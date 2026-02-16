"""Distance metrics for QNM observable space.

This module is intentionally small and dependency-free (stdlib only).

Metrics operate in **log-space**: (ln f, ln Q).

Provided metrics
----------------
- euclidean_log: Euclidean distance in (ln f, ln Q)
- mahalanobis_log: Mahalanobis distance in (ln f, ln Q) with optional
  correlation between ln f and ln Q.

Compatibility / aliasing
------------------------
The project historically used several parameter names.  To keep the
public API stable, `mahalanobis_log` accepts both modern names
(sigma_lnf, sigma_lnQ, r) and legacy aliases (sigma_logf, sigma_logQ,
correlation/rho/corr_logf_logQ, cov_logf_logQ).

All validation is "fail-fast": invalid covariance inputs raise
ValueError("Non-invertible covariance: ...").
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

MetricFn = Callable[..., float]

# ---------------------------------------------------------------------------
# Public constants (kept for compatibility with older callers/tests)
# ---------------------------------------------------------------------------

DEFAULT_SIGMA_LNF: float = 0.07
DEFAULT_SIGMA_LNQ: float = 0.25
DEFAULT_CORRELATION: float = 0.90


def _as_float(x: object, *, name: str) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Non-invertible covariance: {name} must be a float") from e
    return v


def _require_finite_positive(v: float, *, name: str) -> None:
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"Non-invertible covariance: {name} must be finite and > 0")


def euclidean_log(
    lnf_obs: float,
    lnQ_obs: float,
    lnf_atlas: float,
    lnQ_atlas: float,
    **_: object,
) -> float:
    """Euclidean distance in (ln f, ln Q) space."""
    d0 = lnf_obs - lnf_atlas
    d1 = lnQ_obs - lnQ_atlas
    return math.hypot(d0, d1)


def mahalanobis_log(
    lnf_obs: float,
    lnQ_obs: float,
    lnf_atlas: float,
    lnQ_atlas: float,
    *,
    sigma_lnf: Optional[float] = None,
    sigma_lnQ: Optional[float] = None,
    r: Optional[float] = None,
    # legacy/alternate aliases (accepted via **kwargs from callers)
    sigma_logf: Optional[float] = None,
    sigma_logQ: Optional[float] = None,
    rho: Optional[float] = None,
    correlation: Optional[float] = None,
    corr_logf_logQ: Optional[float] = None,
    cov_logf_logQ: Optional[float] = None,
    **_: object,
) -> float:
    """Mahalanobis distance in (ln f, ln Q). Returns d (not d²)."""

    # --- normalize sigmas ---------------------------------------------------
    if sigma_lnf is None:
        sigma_lnf = sigma_logf
    if sigma_lnQ is None:
        sigma_lnQ = sigma_logQ

    if sigma_lnf is None or sigma_lnQ is None:
        raise ValueError("sigma_lnf and sigma_lnQ are required")

    sigma_lnf = _as_float(sigma_lnf, name="sigma_logf")
    sigma_lnQ = _as_float(sigma_lnQ, name="sigma_logQ")
    _require_finite_positive(sigma_lnf, name="sigma_logf")
    _require_finite_positive(sigma_lnQ, name="sigma_logQ")

    # --- normalize correlation / covariance --------------------------------
    if r is None:
        r = correlation if correlation is not None else rho
    if r is None and corr_logf_logQ is not None:
        r = corr_logf_logQ

    if cov_logf_logQ is not None:
        cov = _as_float(cov_logf_logQ, name="cov_logf_logQ")
        if not math.isfinite(cov):
            raise ValueError("Non-invertible covariance: cov_logf_logQ must be finite")
        r = cov / (sigma_lnf * sigma_lnQ)

    if r is None:
        r = DEFAULT_CORRELATION
    r = _as_float(r, name="correlation")
    if not math.isfinite(r) or abs(r) >= 1.0:
        raise ValueError("Non-invertible covariance: |r| must be < 1")

    # --- compute d^2 via closed-form inverse of 2x2 covariance --------------
    d0 = lnf_obs - lnf_atlas
    d1 = lnQ_obs - lnQ_atlas

    s0 = sigma_lnf
    s1 = sigma_lnQ
    cov = r * s0 * s1

    det = (s0 * s0) * (s1 * s1) - (cov * cov)
    if det <= 0.0 or not math.isfinite(det):
        raise ValueError("Non-invertible covariance: det(Σ) <= 0")

    inv00 = (s1 * s1) / det
    inv11 = (s0 * s0) / det
    inv01 = -cov / det

    d2 = d0 * d0 * inv00 + 2.0 * d0 * d1 * inv01 + d1 * d1 * inv11
    if d2 < 0.0 and d2 > -1e-15:
        d2 = 0.0
    return math.sqrt(d2)


def kl_bits(distances: list[float]) -> float:
    """Compute KL divergence (in bits) between posterior and uniform prior."""
    N = len(distances)
    if N <= 1:
        return 0.0

    energies = [0.5 * float(d) * float(d) for d in distances]
    e_min = min(energies)
    ws = [math.exp(-(e - e_min)) for e in energies]
    Z = sum(ws)
    if Z <= 0.0 or not math.isfinite(Z):
        return 0.0

    invZ = 1.0 / Z
    log2N = math.log2(N)
    kl = 0.0
    for w in ws:
        q = w * invZ
        if q <= 0.0:
            continue
        kl += q * (math.log2(q) + log2N)

    if kl < 0.0 and kl > -1e-15:
        kl = 0.0
    return kl


def get_metric(name: str) -> MetricFn:
    if name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}. Available: {sorted(METRICS)}")
    return METRICS[name]


METRICS: Dict[str, MetricFn] = {
    "euclidean_log": euclidean_log,
    "mahalanobis_log": mahalanobis_log,
}
