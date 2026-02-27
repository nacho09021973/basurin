from __future__ import annotations

import math

import numpy as np
import pytest

from mvp.brunete.core import J0, K_R, chi_psd, estimate_s1_kappa_polyfit


def test_psd_plana_chi_y_curvatura_base() -> None:
    f0 = 100.0
    freqs = np.linspace(80.0, 120.0, 401)
    psd = np.ones_like(freqs)

    s1, kappa, _ = estimate_s1_kappa_polyfit(freqs, psd, f0_hz=f0, half_window_hz=10.0)
    assert s1 == pytest.approx(0.0, abs=1e-12)
    assert kappa == pytest.approx(0.0, abs=1e-12)

    q = 4.3
    rho0 = 10.0
    assert chi_psd(q, s1, kappa) == pytest.approx(0.0, abs=1e-14)

    k_val, _ = K_R(rho0=rho0, Q=q, s1=s1, kappa=kappa)
    assert k_val == pytest.approx(-3.0 / (rho0 * rho0), rel=1e-12)


def test_ley_potencia_recupera_s1_kappa_y_chi() -> None:
    alpha = 2.5
    f0 = 180.0
    q = 6.0
    freqs = np.linspace(120.0, 260.0, 1201)
    psd = freqs ** alpha

    s1, kappa, _ = estimate_s1_kappa_polyfit(freqs, psd, f0_hz=f0, half_window_hz=40.0)
    assert s1 == pytest.approx(alpha, rel=1e-10, abs=1e-10)
    assert kappa == pytest.approx(-alpha, rel=1e-10, abs=1e-10)

    expected_chi = abs(alpha * alpha - alpha) / (24.0 * q * q)
    assert chi_psd(q, s1, kappa) == pytest.approx(expected_chi, rel=1e-10)


def test_resummacion_j0_en_cero() -> None:
    assert J0(0.0) == pytest.approx(math.pi / 2.0, rel=0.0, abs=0.0)


def test_determinismo_bit_a_bit() -> None:
    alpha = -1.75
    f0 = 140.0
    freqs = np.linspace(90.0, 210.0, 500)
    psd = freqs ** alpha

    out1 = estimate_s1_kappa_polyfit(freqs, psd, f0_hz=f0, half_window_hz=30.0)
    out2 = estimate_s1_kappa_polyfit(freqs, psd, f0_hz=f0, half_window_hz=30.0)
    assert out1 == out2


def test_error_claro_si_hay_pocos_puntos_en_ventana() -> None:
    freqs = np.array([99.5, 100.0, 100.5, 101.0], dtype=np.float64)
    psd = np.ones_like(freqs)

    with pytest.raises(ValueError, match="al menos 3"):
        estimate_s1_kappa_polyfit(freqs, psd, f0_hz=100.0, half_window_hz=0.1)
