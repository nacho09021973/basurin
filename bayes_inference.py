from __future__ import annotations

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

    x = np.linspace(0.0, 1.0, y.size, dtype=float)
    # deterministic RNG without touching global state.
    rng = np.random.default_rng(seed)

    scores: dict[str, float] = {}
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
        posterior[model] = [float(v) for v in coef.tolist()]

    best_model = max(scores, key=scores.get)
    return {
        "best_model": best_model,
        "model_scores": {k: float(v) for k, v in sorted(scores.items())},
        "posterior_means": posterior,
        "seed": int(seed),
        "n_monte_carlo": int(n_monte_carlo),
    }
