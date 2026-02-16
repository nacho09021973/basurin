from __future__ import annotations

import math
from typing import Any, Callable


def euclidean_log(
    lnf_obs: float,
    lnQ_obs: float,
    lnf_atlas: float,
    lnQ_atlas: float,
    **_: Any,
) -> float:
    """Euclidean distance in (ln f, ln Q)."""
    dlnf = lnf_obs - lnf_atlas
    dlnQ = lnQ_obs - lnQ_atlas
    return math.sqrt(dlnf * dlnf + dlnQ * dlnQ)


def _pick_first(params: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in params and params[key] is not None:
            return params[key]
    return None


def mahalanobis_log(
    lnf_obs: float,
    lnQ_obs: float,
    lnf_atlas: float,
    lnQ_atlas: float,
    *,
    sigma_lnf: float | None = None,
    sigma_lnQ: float | None = None,
    r: float | None = None,
    **kwargs: Any,
) -> float:
    """Mahalanobis distance in (ln f, ln Q), returning d (not d²).

    Supported aliases:
    - sigma_lnf / sigma_logf
    - sigma_lnQ / sigma_logQ
    - r / rho / correlation / corr_logf_logQ
    - cov_logf_logQ (converted to r = cov/(sigma_lnf*sigma_lnQ))
    """
    params = dict(kwargs)

    sigma_lnf = sigma_lnf if sigma_lnf is not None else _pick_first(params, "sigma_lnf", "sigma_logf")
    sigma_lnQ = sigma_lnQ if sigma_lnQ is not None else _pick_first(params, "sigma_lnQ", "sigma_logQ")
    r = r if r is not None else _pick_first(params, "r", "rho", "correlation", "corr_logf_logQ")

    cov = _pick_first(params, "cov_logf_logQ")

    if sigma_lnf is None or sigma_lnQ is None:
        raise ValueError("Non-invertible covariance: sigma_lnf and sigma_lnQ are required")

    sigma_lnf = float(sigma_lnf)
    sigma_lnQ = float(sigma_lnQ)

    if not math.isfinite(sigma_lnf) or sigma_lnf <= 0:
        raise ValueError("Non-invertible covariance: sigma_lnf must be finite and > 0")
    if not math.isfinite(sigma_lnQ) or sigma_lnQ <= 0:
        raise ValueError("Non-invertible covariance: sigma_lnQ must be finite and > 0")

    if cov is not None:
        cov = float(cov)
        if not math.isfinite(cov):
            raise ValueError("Non-invertible covariance: cov_logf_logQ must be finite")
        r = cov / (sigma_lnf * sigma_lnQ)

    if r is None:
        r = 0.0
    r = float(r)
    if not math.isfinite(r) or abs(r) >= 1.0:
        raise ValueError("Non-invertible covariance: |r| must be < 1")

    dlnf = lnf_obs - lnf_atlas
    dlnQ = lnQ_obs - lnQ_atlas

    denom = 1.0 - r * r
    d2 = (
        (dlnf * dlnf) / (sigma_lnf * sigma_lnf)
        + (dlnQ * dlnQ) / (sigma_lnQ * sigma_lnQ)
        - (2.0 * r * dlnf * dlnQ) / (sigma_lnf * sigma_lnQ)
    ) / denom
    if d2 < 0 and d2 > -1e-12:
        d2 = 0.0
    return math.sqrt(d2)


_METRICS: dict[str, Callable[..., float]] = {
    "euclidean_log": euclidean_log,
    "mahalanobis_log": mahalanobis_log,
}


def get_metric(name: str) -> Callable[..., float]:
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric: {name}") from exc
