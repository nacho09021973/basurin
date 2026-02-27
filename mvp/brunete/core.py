"""Core matemático BRUNETE (funciones puras)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


_MIN_POINTS_POLYFIT = 3


def estimate_s1_kappa_polyfit(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    f0_hz: float,
    half_window_hz: float,
) -> tuple[float, float, dict[str, Any]]:
    """Estima ``s1`` y ``kappa`` vía ajuste local cuadrático de ``ln S``.

    Se ajusta ``L(Δu)=a2·Δu²+a1·Δu+a0`` con ``Δu = ln(f)-ln(f0)``.
    Entonces:
      - ``s1 = dL/du|f0 = a1``
      - ``kappa = d²L/du² - s1 = 2*a2 - s1``
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    s = np.asarray(psd, dtype=np.float64)

    if f.ndim != 1 or s.ndim != 1 or f.shape != s.shape:
        raise ValueError("freqs_hz y psd deben ser arrays 1D de igual longitud")
    if f0_hz <= 0.0:
        raise ValueError("f0_hz debe ser > 0")
    if half_window_hz <= 0.0:
        raise ValueError("half_window_hz debe ser > 0")
    if np.any(f <= 0.0) or np.any(s <= 0.0):
        raise ValueError("freqs_hz y psd deben ser estrictamente positivos")

    mask = np.abs(f - f0_hz) <= half_window_hz
    n_points = int(np.count_nonzero(mask))
    if n_points < _MIN_POINTS_POLYFIT:
        raise ValueError(
            "Puntos insuficientes en ventana para ajuste grado 2: "
            f"se requieren al menos {_MIN_POINTS_POLYFIT}, encontrados {n_points} "
            f"(f0_hz={f0_hz}, half_window_hz={half_window_hz})"
        )

    f_win = f[mask]
    s_win = s[mask]
    u0 = math.log(f0_hz)
    du = np.log(f_win) - u0
    ln_s = np.log(s_win)

    a2, a1, a0 = np.polyfit(du, ln_s, deg=2)
    s1 = float(a1)
    d2l_du2 = float(2.0 * a2)
    kappa = float(d2l_du2 - s1)

    meta = {
        "n_points": n_points,
        "f0_hz": float(f0_hz),
        "half_window_hz": float(half_window_hz),
        "window_min_hz": float(f_win.min()),
        "window_max_hz": float(f_win.max()),
        "poly_coeffs": (float(a2), float(a1), float(a0)),
    }
    return s1, kappa, meta


def sigma(Q: float, kappa: float) -> float:
    """Parámetro de control sigma = kappa / (8 Q^2)."""
    return float(kappa / (8.0 * Q * Q))


def chi_psd(Q: float, s1: float, kappa: float) -> float:
    """Diagnóstico chi_PSD = |s1^2 + kappa| / (24 Q^2)."""
    return float(abs(s1 * s1 + kappa) / (24.0 * Q * Q))


def K_R(rho0: float, Q: float, s1: float, kappa: float) -> tuple[float, float]:
    """Teorema 1: K = -3/rho0^2 * (1 - (s1^2 + kappa)/(24Q^2)), R=2K."""
    k_val = -3.0 / (rho0 * rho0) * (1.0 - (s1 * s1 + kappa) / (24.0 * Q * Q))
    r_val = 2.0 * k_val
    return float(k_val), float(r_val)


def _erfcx_stable(x: float) -> float:
    """erfcx(x)=exp(x^2)*erfc(x) estable para x grande."""
    if x < 25.0:
        return math.exp(x * x) * math.erfc(x)
    inv = 1.0 / x
    inv2 = inv * inv
    return inv / math.sqrt(math.pi) * (1.0 + 0.5 * inv2 + 0.75 * inv2 * inv2)


def J0(sigma_value: float) -> float:
    """Integral regularizada J0(sigma) (forma cerrada con erfc), sigma >= 0."""
    if sigma_value < 0.0:
        raise ValueError("J0 solo está definida para sigma >= 0")
    if sigma_value == 0.0:
        return math.pi / 2.0

    sqrt_sigma = math.sqrt(sigma_value)
    exp_erfc = _erfcx_stable(sqrt_sigma)
    return float(
        math.pi
        * ((sigma_value + 0.5) * exp_erfc - math.sqrt(sigma_value / math.pi))
    )


def J1(sigma_value: float) -> float:
    """J1(sigma) usando la identidad J1 = -dJ0/dsigma."""
    if sigma_value < 0.0:
        raise ValueError("J1 solo está definida para sigma >= 0")
    if sigma_value == 0.0:
        return math.pi / 2.0

    sqrt_sigma = math.sqrt(sigma_value)
    exp_erfc = _erfcx_stable(sqrt_sigma)
    return float(
        math.pi
        * (
            (sigma_value + 1.0) / (math.sqrt(math.pi) * sqrt_sigma)
            - (sigma_value + 1.5) * exp_erfc
        )
    )
