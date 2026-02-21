from __future__ import annotations

import math

from mvp.oracle_t0_math import compute_central_z_grad, compute_local_cv_max


def _window(ln_f: float, ln_q: float, sigma_f: float = 0.1, sigma_q: float = 0.1) -> dict[str, float]:
    return {
        "ln_f_220": ln_f,
        "ln_Q_220": ln_q,
        "sigma_ln_f_220": sigma_f,
        "sigma_ln_Q_220": sigma_q,
    }


def test_plateau_perfecto_z_grad_y_cv_casi_cero() -> None:
    windows = [_window(4.6, 2.1) for _ in range(7)]

    z_grad = compute_central_z_grad(windows)
    cv_max = compute_local_cv_max(windows, w=3)

    assert z_grad[0] is None and z_grad[-1] is None
    assert all((v is None) or v == 0.0 for v in z_grad)
    assert all((v is not None) and math.isclose(v, 0.0, abs_tol=1e-12) for v in cv_max)


def test_salto_genera_z_grad_alto_en_torno_al_salto() -> None:
    windows = [
        _window(4.0, 2.0),
        _window(4.0, 2.0),
        _window(4.0, 2.0),
        _window(6.0, 3.5),
        _window(6.0, 3.5),
        _window(6.0, 3.5),
    ]

    z_grad = compute_central_z_grad(windows)

    assert z_grad[2] is not None and z_grad[2] > 10.0
    assert z_grad[3] is not None and z_grad[3] > 10.0


def test_sigma_cero_no_explota_usa_sigma_floor() -> None:
    windows = [
        _window(1.0, 1.0, sigma_f=0.0, sigma_q=0.0),
        _window(1.2, 1.3, sigma_f=0.0, sigma_q=0.0),
        _window(1.4, 1.6, sigma_f=0.0, sigma_q=0.0),
    ]

    z_grad = compute_central_z_grad(windows, sigma_floor_ln_f=0.05, sigma_floor_ln_q=0.05)

    assert z_grad[1] is not None
    assert math.isfinite(z_grad[1])
    assert z_grad[1] > 0.0


def test_mediana_casi_cero_no_explota_usa_scale_floor() -> None:
    windows = [
        _window(-1e-7, 1e-7),
        _window(0.0, 0.0),
        _window(1e-7, -1e-7),
    ]

    cv_max = compute_local_cv_max(windows, w=1, scale_floor_ln_f=1e-3, scale_floor_ln_q=1e-3)

    assert cv_max[0] is None and cv_max[2] is None
    assert cv_max[1] is not None
    assert math.isfinite(cv_max[1])
    assert cv_max[1] >= 0.0
