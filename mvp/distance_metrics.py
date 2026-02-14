<<<<<<< HEAD
=======
"""Distance metrics and information measures for QNM observable space.

Metrics
-------
1. **euclidean_log** (current default):
   d = sqrt[(ln f_obs - ln f_th)² + (ln Q_obs - ln Q_th)²]
   Assumes isotropic, uncorrelated errors.  Simple but naive.

2. **mahalanobis_log** (Fisher-informed):
   d² = 1/(1-r²) * [(Δf/σ_f)² + (ΔQ/σ_Q)² - 2r·(Δf/σ_f)·(ΔQ/σ_Q)]
   where Δf = ln f_obs - ln f_th,  ΔQ = ln Q_obs - ln Q_th,
   σ_f, σ_Q are fractional uncertainties, and r is the f-Q correlation.

   The correlation r ≈ 0.9 comes from the Fisher information matrix of a
   damped sinusoid in Gaussian noise (Berti et al. 2006).  The elliptical
   iso-distance contours capture the physical fact that f and Q are strongly
   correlated in ringdown parameter estimation.

Information measure
-------------------
**kl_bits(distances)** — KL divergence D_KL(posterior ‖ prior) in bits.

The old measure log₂(N/n) assumes a uniform prior and a hard binary cut
at epsilon.  KL divergence instead uses the *full* distance distribution:

    L(gᵢ) ∝ exp(−dᵢ²/2)        (Gaussian likelihood from distance)
    P(gᵢ|data) = L(gᵢ) / Z     (posterior)
    P(gᵢ) = 1/N                 (uniform prior)
    D_KL = Σᵢ P(gᵢ|data) · log₂[ P(gᵢ|data) / P(gᵢ) ]

This is epsilon-independent, uses the full ranking, and naturally handles
the Mahalanobis metric (where d is already in σ units, so exp(−d²/2) is
the proper Gaussian likelihood).

References:
  - Berti, Cardoso & Will (2006) PRD 73, 064030 (Fisher matrix for ringdown)
  - Gemini research compilation for BASURIN (2025)
"""
>>>>>>> 362cb04 (Add KL divergence (bits_kl) as epsilon-independent information measure)
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


<<<<<<< HEAD
def get_metric(name: str) -> Callable[..., float]:
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric: {name}") from exc
=======
def get_metric(name: str) -> MetricFn:
    """Look up a metric function by name.

    Raises ValueError if not found.
    """
    if name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}. Available: {list(METRICS)}")
    return METRICS[name]


# ---------------------------------------------------------------------------
# Information measure: KL divergence in bits
# ---------------------------------------------------------------------------

def _logsumexp(xs: list[float]) -> float:
    """Numerically stable log-sum-exp (avoids overflow/underflow)."""
    if not xs:
        return float("-inf")
    x_max = max(xs)
    if x_max == float("-inf"):
        return float("-inf")
    return x_max + math.log(sum(math.exp(x - x_max) for x in xs))


def kl_bits(distances: list[float]) -> float:
    """KL divergence D_KL(posterior ‖ uniform prior) in bits.

    Parameters
    ----------
    distances : list of float
        Non-negative distances from the observation to each atlas entry.

    Returns
    -------
    float
        Information gained (in bits) by observing (f, Q).
        Always >= 0.  Returns 0.0 for empty or single-entry atlas.

    Notes
    -----
    Likelihood:  L_i = exp(-d_i² / 2)
    Posterior:   P_i = L_i / Z,  Z = Σ L_j
    Prior:       Q_i = 1/N  (uniform)
    D_KL = Σ P_i · log₂(P_i / Q_i)
         = Σ P_i · log₂(N · P_i)
         = log₂(N) + Σ P_i · log₂(P_i)
         = log₂(N) - H(posterior)

    So D_KL = log₂(N) minus the Shannon entropy of the posterior.
    When the posterior concentrates on one geometry, D_KL → log₂(N).
    When the posterior is uniform (all d equal), D_KL → 0.
    """
    n = len(distances)
    if n <= 1:
        return 0.0

    # log-likelihood: log L_i = -d_i² / 2
    log_likes = [-0.5 * d * d for d in distances]

    # log Z = logsumexp(log_likes)
    log_z = _logsumexp(log_likes)

    # log posterior: log P_i = log L_i - log Z
    # D_KL = log₂(N) + Σ P_i · log₂(P_i)
    #       = log₂(N) + (1/ln2) · Σ P_i · ln(P_i)
    # where ln(P_i) = log_likes[i] - log_z
    log2_n = math.log2(n)
    neg_entropy_nats = 0.0
    for ll in log_likes:
        log_p = ll - log_z  # ln P_i
        p = math.exp(log_p)  # P_i
        if p > 0:
            neg_entropy_nats += p * log_p  # Σ P_i · ln(P_i)

    # Convert nats → bits: divide by ln(2)
    neg_entropy_bits = neg_entropy_nats / math.log(2)
    kl = log2_n + neg_entropy_bits  # D_KL = log₂(N) - H(posterior)

    # Guard against floating-point drift below zero
    return max(0.0, kl)
>>>>>>> 362cb04 (Add KL divergence (bits_kl) as epsilon-independent information measure)
