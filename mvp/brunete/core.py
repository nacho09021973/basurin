"""Core matemático BRUNETE (funciones puras)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


_NOT_APPLICABLE = "not_applicable"


def psd_log_derivatives_polyfit(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    f0_hz: float,
    half_window_hz: float,
    min_points: int,
) -> tuple[float, float, dict[str, Any]]:
    """Estima ``s1`` y ``kappa`` vía ajuste local cuadrático de ``ln S``.

    Ajuste: ``L(Δu)=a2*Δu^2+a1*Δu+a0`` con ``Δu = ln(f)-ln(f0)``.
    Identidades BRUNETE: ``s1=dL/du`` y ``kappa=d²L/du² - s1``.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    s = np.asarray(psd, dtype=np.float64)

    if f.ndim != 1 or s.ndim != 1 or f.shape != s.shape:
        raise ValueError("freqs_hz y psd deben ser arrays 1D de igual longitud")
    if f0_hz <= 0.0:
        raise ValueError("f0_hz debe ser > 0")
    if half_window_hz <= 0.0:
        raise ValueError("half_window_hz debe ser > 0")
    if min_points < 3:
        raise ValueError("min_points debe ser >= 3 para ajuste polinomial de grado 2")
    if np.any(f <= 0.0) or np.any(s <= 0.0):
        raise ValueError("freqs_hz y psd deben ser estrictamente positivos")

    mask = np.abs(f - f0_hz) <= half_window_hz
    n_points = int(np.count_nonzero(mask))
    if n_points < min_points:
        raise ValueError(
            "Puntos insuficientes en ventana para ajuste grado 2: "
            f"se requieren al menos {min_points}, encontrados {n_points} "
            f"(f0_hz={f0_hz}, half_window_hz={half_window_hz})"
        )

    f_win = f[mask]
    s_win = s[mask]
    du = np.log(f_win) - math.log(f0_hz)
    ln_s = np.log(s_win)

    a2, a1, a0 = np.polyfit(du, ln_s, deg=2)
    s1 = float(a1)
    kappa = float(2.0 * a2 - s1)

    meta = {
        "n_points": n_points,
        "f0_hz": float(f0_hz),
        "half_window_hz": float(half_window_hz),
        "poly_coeffs": (float(a2), float(a1), float(a0)),
        "window_min_hz": float(f_win.min()),
        "window_max_hz": float(f_win.max()),
    }
    return s1, kappa, meta


def sigma(Q: float, kappa: float) -> float:
    """Parámetro de control BRUNETE: ``sigma = kappa/(8 Q^2)``."""
    return float(kappa / (8.0 * Q * Q))


def chi_psd(Q: float, s1: float, kappa: float) -> float:
    """Diagnóstico BRUNETE: ``chi_PSD = |s1^2 + kappa|/(24 Q^2)``."""
    return float(abs(s1 * s1 + kappa) / (24.0 * Q * Q))


def curvature_KR(rho0: float, Q: float, s1: float, kappa: float) -> tuple[float, float]:
    """Teorema 1 BRUNETE: ``K`` y ``R=2K``."""
    K = -3.0 / (rho0 * rho0) * (1.0 - (s1 * s1 + kappa) / (24.0 * Q * Q))
    R = 2.0 * K
    return float(K), float(R)


def _erfcx_stable(x: float) -> float:
    if x < 25.0:
        return math.exp(x * x) * math.erfc(x)
    inv = 1.0 / x
    inv2 = inv * inv
    inv4 = inv2 * inv2
    return inv / math.sqrt(math.pi) * (1.0 - 0.5 * inv2 + 0.75 * inv4 - 1.875 * inv4 * inv2)


def _j0_closed_form_nonnegative(sigma_value: float) -> float:
    if sigma_value == 0.0:
        return math.pi / 2.0
    sqrt_sigma = math.sqrt(sigma_value)
    erfcx = _erfcx_stable(sqrt_sigma)
    return float(
        math.pi
        * ((sigma_value + 0.5) * erfcx - math.sqrt(sigma_value / math.pi))
    )


def _j1_closed_form_nonnegative(sigma_value: float) -> float:
    if sigma_value == 0.0:
        return math.pi / 2.0
    sqrt_sigma = math.sqrt(sigma_value)
    erfcx = _erfcx_stable(sqrt_sigma)
    return float(
        math.pi
        * (
            (sigma_value + 1.0) / (math.sqrt(math.pi) * sqrt_sigma)
            - (sigma_value + 1.5) * erfcx
        )
    )


def J0_J1(sigma_value: float, sigma_switch: float = 0.1) -> tuple[float | None, float | None, dict[str, Any]]:
    """Evalúa ``𝓙0`` y ``𝓙1`` con ramas perturbativa/cerrada y contrato para sigma<0.

    * ``|sigma| < sigma_switch``: usa (6.8)-(6.9): ``𝓙0≈π/2*(1-sigma)``, ``𝓙1≈π/2``.
    * ``sigma >= sigma_switch``: usa forma cerrada (6.7)/(A.4) y ``𝓙1=-d𝓙0/dsigma``.
    * ``sigma < 0`` y ``|sigma| >= sigma_switch``: ``not_applicable``.
    """
    if sigma_switch <= 0.0:
        raise ValueError("sigma_switch debe ser > 0")

    if abs(sigma_value) < sigma_switch:
        return (
            float((math.pi / 2.0) * (1.0 - sigma_value)),
            float(math.pi / 2.0),
            {"mode": "perturbative", "status": "ok"},
        )

    if sigma_value < 0.0:
        return None, None, {"mode": "closed_form", "status": _NOT_APPLICABLE}

    return (
        _j0_closed_form_nonnegative(sigma_value),
        _j1_closed_form_nonnegative(sigma_value),
        {"mode": "closed_form", "status": "ok"},
    )


# Compatibilidad hacia atrás en este MVP.
def estimate_s1_kappa_polyfit(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    f0_hz: float,
    half_window_hz: float,
) -> tuple[float, float, dict[str, Any]]:
    return psd_log_derivatives_polyfit(
        freqs_hz=freqs_hz,
        psd=psd,
        f0_hz=f0_hz,
        half_window_hz=half_window_hz,
        min_points=3,
    )


def K_R(rho0: float, Q: float, s1: float, kappa: float) -> tuple[float, float]:
    return curvature_KR(rho0=rho0, Q=Q, s1=s1, kappa=kappa)


def J0(sigma_value: float) -> float:
    """Evalúa ``𝓙0(σ)`` usando exclusivamente la forma cerrada (A.4/6.7).

    La aproximación asintótica de A.5 *no* es normativa para implementación.
    """
    if sigma_value < 0.0:
        raise ValueError("J0 solo está definida para sigma >= 0")
    return _j0_closed_form_nonnegative(sigma_value)


def J1(sigma_value: float) -> float:
    if sigma_value < 0.0:
        raise ValueError("J1 solo está definida para sigma >= 0")
    return _j1_closed_form_nonnegative(sigma_value)
