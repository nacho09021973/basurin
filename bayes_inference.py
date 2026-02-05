from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BayesParams:
    seed: int = 42
    n_monte_carlo: int = 500
    prior_precision: float = 1e-6


def _design_matrix(x: np.ndarray, model: str) -> np.ndarray:
    if model == "linear":
        return np.column_stack([np.ones_like(x), x])
    if model == "poly2":
        return np.column_stack([np.ones_like(x), x, x**2])
    raise ValueError(f"unsupported model: {model}")


def run_bayes_model_selection(
    values: Iterable[float],
    models: list[str],
    *,
    seed: int,
    n_monte_carlo: int,
    k_features: int,
    prior_precision: float,
) -> dict[str, object]:
    y = np.asarray(list(values), dtype=float)
    if y.size == 0:
        raise ValueError("values must not be empty")

    n = y.size
    x = np.linspace(0.0, 1.0, n, dtype=float)
    # deterministic RNG without touching global state.
    rng = np.random.default_rng(seed)

    scores: dict[str, float] = {}
    bic_values: dict[str, float] = {}
    posterior: dict[str, list[float]] = {}

    for model in models:
        phi = _design_matrix(x, model)
        k = min(int(k_features), phi.shape[1])
        phi_k = phi[:, :k]

        gram = phi_k.T @ phi_k + float(prior_precision) * np.eye(k)
        rhs = phi_k.T @ y
        coef = np.linalg.solve(gram, rhs)

        noise_scale = max(1e-12, float(np.std(y - phi_k @ coef)))
        samples = rng.normal(loc=coef, scale=noise_scale, size=(int(n_monte_carlo), k))

        pred = phi_k @ samples.T
        mse = float(np.mean((pred.mean(axis=1) - y) ** 2))
        scores[model] = -mse

        # BIC = n*ln(MSE) + k*ln(n)  (Schwarz criterion, Gaussian errors)
        bic_values[model] = n * math.log(max(mse, 1e-300)) + k * math.log(n)

        posterior[model] = [float(v) for v in coef.tolist()]

    best_model = max(scores, key=scores.get)

    # delta_score: gap between best and second-best (in score's natural order)
    sorted_scores = sorted(scores.values(), reverse=True)
    delta_score = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0

    # BIC-based Bayes Factor proxy: log BF(best vs second) ≈ -½(BIC_best - BIC_second)
    # Lower BIC is better; positive log_bf means evidence for the BIC-best model.
    bic_ranked = sorted(bic_values.items(), key=lambda item: item[1])
    bic_best_model = bic_ranked[0][0]

    log_bf_proxy: float | None = None
    bf_method: str | None = None
    if len(bic_ranked) >= 2:
        log_bf_proxy = -0.5 * (bic_ranked[0][1] - bic_ranked[1][1])
        bf_method = "bic_schwarz"

    return {
        "best_model": best_model,
        "model_scores": {mk: float(mv) for mk, mv in sorted(scores.items())},
        "score_name": "neg_mse",
        "score_higher_is_better": True,
        "delta_score": float(delta_score),
        "bic": {mk: float(mv) for mk, mv in sorted(bic_values.items())},
        "log_bayes_factor_proxy": float(log_bf_proxy) if log_bf_proxy is not None else None,
        "bf_proxy_method": bf_method,
        "selection_consistent": best_model == bic_best_model,
        "posterior_means": posterior,
        "seed": int(seed),
        "n_monte_carlo": int(n_monte_carlo),
    }
