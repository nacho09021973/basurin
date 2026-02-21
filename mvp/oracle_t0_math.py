from __future__ import annotations

import math
from statistics import median
from typing import Any

# Default conservador: evita inflar z por incertidumbres degeneradas (sigma -> 0)
# sin aplanar excesivamente gradientes reales en escala logarítmica.
DEFAULT_SIGMA_FLOOR_LN_F = 1e-3
DEFAULT_SIGMA_FLOOR_LN_Q = 1e-3

# Default pequeño para evitar división por cero cuando mediana ~ 0 en CV.
DEFAULT_SCALE_FLOOR_LN_F = 1e-6
DEFAULT_SCALE_FLOOR_LN_Q = 1e-6

DEFAULT_PLATEAU_HALFWIDTH_STEPS = 3
_IQR_TO_SIGMA = 1.349


def sigma_from_iqr(iqr: float) -> float:
    """Convierte IQR a sigma robusta usando sigma ~= IQR / 1.349."""

    return float(iqr) / _IQR_TO_SIGMA


def _safe_sigma(value: Any, *, floor: float) -> float:
    try:
        sigma = float(value)
    except (TypeError, ValueError):
        sigma = float("nan")

    if not math.isfinite(sigma) or sigma <= 0.0:
        return float(floor)
    return max(float(floor), sigma)


def _sigma_for_window(window: dict[str, Any], *, sigma_key: str, iqr_key: str, floor: float) -> float:
    sigma = window.get(sigma_key)
    if sigma is None:
        iqr_value = window.get(iqr_key)
        if iqr_value is not None:
            sigma = sigma_from_iqr(float(iqr_value))
    return _safe_sigma(sigma, floor=floor)


def compute_central_z_grad(
    windows: list[dict[str, Any]],
    *,
    sigma_floor_ln_f: float = DEFAULT_SIGMA_FLOOR_LN_F,
    sigma_floor_ln_q: float = DEFAULT_SIGMA_FLOOR_LN_Q,
) -> list[float | None]:
    """Calcula z_grad por diferencia central para k=1..N-2 y null en bordes."""

    n = len(windows)
    out: list[float | None] = [None] * n
    if n < 3:
        return out

    ln_f = [float(w["ln_f_220"]) for w in windows]
    ln_q = [float(w["ln_Q_220"]) for w in windows]
    sig_f = [
        _sigma_for_window(
            w,
            sigma_key="sigma_ln_f_220",
            iqr_key="iqr_ln_f_220",
            floor=sigma_floor_ln_f,
        )
        for w in windows
    ]
    sig_q = [
        _sigma_for_window(
            w,
            sigma_key="sigma_ln_Q_220",
            iqr_key="iqr_ln_Q_220",
            floor=sigma_floor_ln_q,
        )
        for w in windows
    ]

    for k in range(1, n - 1):
        denom_f = math.sqrt(sig_f[k + 1] ** 2 + sig_f[k - 1] ** 2)
        denom_q = math.sqrt(sig_q[k + 1] ** 2 + sig_q[k - 1] ** 2)

        dz_f = abs(ln_f[k + 1] - ln_f[k - 1]) / denom_f
        dz_q = abs(ln_q[k + 1] - ln_q[k - 1]) / denom_q
        out[k] = math.sqrt(dz_f**2 + dz_q**2)

    return out


def _percentile_linear(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _iqr(values: list[float]) -> float:
    return _percentile_linear(values, 0.75) - _percentile_linear(values, 0.25)


def compute_local_cv_max(
    windows: list[dict[str, Any]],
    *,
    w: int = DEFAULT_PLATEAU_HALFWIDTH_STEPS,
    scale_floor_ln_f: float = DEFAULT_SCALE_FLOOR_LN_F,
    scale_floor_ln_q: float = DEFAULT_SCALE_FLOOR_LN_Q,
) -> list[float | None]:
    """Calcula cv_max robusto por vecindad [k-w, k+w]."""

    n = len(windows)
    out: list[float | None] = [None] * n
    ln_f = [float(win["ln_f_220"]) for win in windows]
    ln_q = [float(win["ln_Q_220"]) for win in windows]

    for k in range(n):
        lo = max(0, k - w)
        hi = min(n - 1, k + w)
        f_nei = ln_f[lo : hi + 1]
        q_nei = ln_q[lo : hi + 1]

        if len(f_nei) < 3:
            out[k] = None
            continue

        denom_f = max(abs(median(f_nei)), float(scale_floor_ln_f))
        denom_q = max(abs(median(q_nei)), float(scale_floor_ln_q))

        cv_f = sigma_from_iqr(_iqr(f_nei)) / denom_f
        cv_q = sigma_from_iqr(_iqr(q_nei)) / denom_q
        out[k] = max(cv_f, cv_q)

    return out


def compute_oracle_t0_math(
    windows: list[dict[str, Any]],
    *,
    sigma_floor_ln_f: float = DEFAULT_SIGMA_FLOOR_LN_F,
    sigma_floor_ln_q: float = DEFAULT_SIGMA_FLOOR_LN_Q,
    scale_floor_ln_f: float = DEFAULT_SCALE_FLOOR_LN_F,
    scale_floor_ln_q: float = DEFAULT_SCALE_FLOOR_LN_Q,
    w: int = DEFAULT_PLATEAU_HALFWIDTH_STEPS,
) -> list[dict[str, Any]]:
    """Enriquece cada ventana con métricas puras del core matemático t0 v1.2."""

    z_grad = compute_central_z_grad(
        windows,
        sigma_floor_ln_f=sigma_floor_ln_f,
        sigma_floor_ln_q=sigma_floor_ln_q,
    )
    cv_max = compute_local_cv_max(
        windows,
        w=w,
        scale_floor_ln_f=scale_floor_ln_f,
        scale_floor_ln_q=scale_floor_ln_q,
    )

    return [
        {
            "index": idx,
            "t0_ms": window.get("t0_ms"),
            "z_grad": z_grad[idx],
            "cv_max": cv_max[idx],
        }
        for idx, window in enumerate(windows)
    ]
