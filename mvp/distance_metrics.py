from __future__ import annotations

import math
from typing import Any


def mahalanobis_log(
    lnf_obs: float,
    lnQ_obs: float,
    lnf_atlas: float,
    lnQ_atlas: float,
    *,
    sigma_lnf: float | None = 1.0,
    sigma_lnQ: float | None = 1.0,
    r: float | None = 0.0,
    **kwargs: Any,
) -> float:
    """
    Mahalanobis distance in (ln f, ln Q) with correlation r.
    Returns d = sqrt(d^2).

    Accepted aliases via **kwargs:
      - sigma_logf -> sigma_lnf
      - sigma_logQ -> sigma_lnQ
      - rho/correlation/corr_logf_logQ -> r
      - cov_logf_logQ: treated as covariance; converted to r = cov/(σf σQ)
    """
    if sigma_lnf is None:
        sigma_lnf = kwargs.get("sigma_logf")
    if sigma_lnQ is None:
        sigma_lnQ = kwargs.get("sigma_logQ")

    if r is None:
        r = (
            kwargs.get("correlation")
            if "correlation" in kwargs
            else kwargs.get("rho", kwargs.get("corr_logf_logQ"))
        )

    if "cov_logf_logQ" in kwargs:
        cov = float(kwargs["cov_logf_logQ"])
        if sigma_lnf is None or sigma_lnQ is None:
            raise ValueError(
                "Non-invertible covariance: sigma_lnf and sigma_lnQ required when cov_logf_logQ is provided"
            )
        r = cov / (float(sigma_lnf) * float(sigma_lnQ))

    sigma_lnf = float(sigma_lnf)  # type: ignore[arg-type]
    sigma_lnQ = float(sigma_lnQ)  # type: ignore[arg-type]
    r = float(r) if r is not None else 0.0

    if not math.isfinite(sigma_lnf) or sigma_lnf <= 0:
        raise ValueError("Non-invertible covariance: sigma_lnf must be finite and > 0")
    if not math.isfinite(sigma_lnQ) or sigma_lnQ <= 0:
        raise ValueError("Non-invertible covariance: sigma_lnQ must be finite and > 0")
    if not math.isfinite(r) or abs(r) >= 1.0:
        raise ValueError("Non-invertible covariance: |r| must be < 1")

    dlnf = lnf_obs - lnf_atlas
    dlnQ = lnQ_obs - lnQ_atlas

    denom = 1.0 - r * r
    num = (dlnf * dlnf) / (sigma_lnf * sigma_lnf) + (dlnQ * dlnQ) / (sigma_lnQ * sigma_lnQ) - (2.0 * r * dlnf * dlnQ) / (sigma_lnf * sigma_lnQ)
    d2 = num / denom
    if d2 < 0 and d2 > -1e-12:
        d2 = 0.0
    return math.sqrt(d2)
