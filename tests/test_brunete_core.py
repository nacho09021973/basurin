from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from mvp.brunete.core import J0, J0_J1, chi_psd, curvature_KR, psd_log_derivatives_polyfit


def test_psd_plana_chi_y_curvatura_base() -> None:
    f0 = 100.0
    freqs = np.linspace(80.0, 120.0, 401)
    psd = np.ones_like(freqs)

    s1, kappa, _ = psd_log_derivatives_polyfit(
        freqs, psd, f0_hz=f0, half_window_hz=10.0, min_points=21
    )
    assert s1 == pytest.approx(0.0, abs=1e-12)
    assert kappa == pytest.approx(0.0, abs=1e-12)

    q = 4.3
    rho0 = 10.0
    assert chi_psd(q, s1, kappa) == pytest.approx(0.0, abs=1e-14)

    K, _ = curvature_KR(rho0=rho0, Q=q, s1=s1, kappa=kappa)
    assert K == pytest.approx(-3.0 / (rho0 * rho0), rel=1e-12)


def test_ley_potencia_recupera_s1_kappa_y_chi() -> None:
    alpha = 2.5
    f0 = 180.0
    q = 6.0
    freqs = np.linspace(120.0, 260.0, 1201)
    psd = freqs ** alpha

    s1, kappa, _ = psd_log_derivatives_polyfit(
        freqs, psd, f0_hz=f0, half_window_hz=40.0, min_points=51
    )
    assert s1 == pytest.approx(alpha, rel=1e-10, abs=1e-10)
    assert kappa == pytest.approx(-alpha, rel=1e-10, abs=1e-10)

    expected_chi = abs(alpha * alpha - alpha) / (24.0 * q * q)
    assert chi_psd(q, s1, kappa) == pytest.approx(expected_chi, rel=1e-10)


def test_resummacion_j0_en_cero() -> None:
    j0, j1, meta = J0_J1(0.0)
    assert meta["status"] == "ok"
    assert j0 == pytest.approx(math.pi / 2.0)
    assert j1 == pytest.approx(math.pi / 2.0)


def test_j0_forma_cerrada_es_source_of_truth() -> None:
    sigma = 3.5
    expected = math.pi * (
        (sigma + 0.5) * math.exp(sigma) * math.erfc(math.sqrt(sigma))
        - math.sqrt(sigma / math.pi)
    )
    assert J0(sigma) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_j0_finita_no_negativa_y_monotona_para_sigma_no_negativo() -> None:
    sigmas = np.linspace(0.0, 25.0, 251)
    values = np.array([J0(float(s)) for s in sigmas], dtype=np.float64)

    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    # sanity check de estabilidad: J0 decrece al aumentar sigma
    assert np.all(np.diff(values) <= 1e-12)


def test_resummacion_sigma_negativo_grande_es_not_applicable() -> None:
    j0, j1, meta = J0_J1(-0.5, sigma_switch=0.1)
    assert j0 is None
    assert j1 is None
    assert meta["status"] == "not_applicable"


def test_determinismo_mismas_entradas_mismas_salidas() -> None:
    alpha = -1.75
    f0 = 140.0
    freqs = np.linspace(90.0, 210.0, 500)
    psd = freqs ** alpha

    out1 = psd_log_derivatives_polyfit(
        freqs, psd, f0_hz=f0, half_window_hz=30.0, min_points=31
    )
    out2 = psd_log_derivatives_polyfit(
        freqs, psd, f0_hz=f0, half_window_hz=30.0, min_points=31
    )
    assert out1 == out2


def test_error_claro_si_no_se_cumple_min_points() -> None:
    freqs = np.array([99.5, 100.0, 100.5, 101.0], dtype=np.float64)
    psd = np.ones_like(freqs)

    with pytest.raises(ValueError, match="se requieren al menos 4"):
        psd_log_derivatives_polyfit(
            freqs, psd, f0_hz=100.0, half_window_hz=0.1, min_points=4
        )


def test_a5_asintotico_inconsistente_con_forma_cerrada_source_of_truth() -> None:
    j0_200, _, _ = J0_J1(200.0, sigma_switch=0.1)
    j0_400, _, _ = J0_J1(400.0, sigma_switch=0.1)
    assert j0_200 is not None and j0_400 is not None

    ratio_closed = j0_400 / j0_200
    ratio_a5 = (math.pi / (2.0 * 400.0)) / (math.pi / (2.0 * 200.0))

    assert ratio_closed == pytest.approx(2.0 ** (-1.5), rel=5e-2)
    assert ratio_closed != pytest.approx(ratio_a5, rel=1e-1)


@pytest.mark.parametrize("sigma_value", [0.1, 1.0, 10.0, 100.0, 1000.0])
def test_erfcx_asintotica_j0_j1_finitas_y_no_negativas(sigma_value: float) -> None:
    j0, j1, meta = J0_J1(sigma_value, sigma_switch=0.1)

    assert meta["status"] == "ok"
    assert j0 is not None and j1 is not None
    assert math.isfinite(j0)
    assert math.isfinite(j1)
    assert j0 >= 0.0
    assert j1 >= 0.0
