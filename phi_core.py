"""Core pure functions for the phenomenological Phi bridge (QNM <-> bulk).

This module is intentionally side-effect free: no IO, no stage contracts, no BASURIN run wiring.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def forward_model(M2_0: float, r1: float, L: float, alpha: float = 1.0) -> dict:
    """Map atlas quantities to predicted ringdown observables.

    Returns keys: ``f``, ``tau``, ``Q``, ``omega_0``, ``gamma``.
    """
    if M2_0 <= 0:
        raise ValueError("M2_0 must be > 0")
    if L <= 0:
        raise ValueError("L must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if r1 <= 1:
        raise ValueError("r1 must be > 1")

    omega_0 = math.sqrt(M2_0) / L
    f_pred = omega_0 / (2.0 * math.pi)
    gamma = alpha * omega_0 * (r1 - 1.0)
    tau_pred = 1.0 / gamma
    Q_pred = math.pi * f_pred * tau_pred

    return {
        "f": f_pred,
        "tau": tau_pred,
        "Q": Q_pred,
        "omega_0": omega_0,
        "gamma": gamma,
    }


def inverse_model(f_obs: float, tau_obs: float, alpha: float = 1.0) -> dict:
    """Map observed ringdown quantities back to ratio-space prediction."""
    if f_obs <= 0:
        raise ValueError("f_obs must be > 0")
    if tau_obs <= 0:
        raise ValueError("tau_obs must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    Q_obs = math.pi * f_obs * tau_obs
    if Q_obs <= 0:
        raise ValueError("Q_obs must be > 0")

    r1_pred = 1.0 + 1.0 / (alpha * 2.0 * Q_obs)
    return {"Q_obs": Q_obs, "r1_pred": r1_pred}


def _as_positive_1d(arr: Iterable[float], *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1)
    if np.any(out <= 0):
        raise ValueError(f"{name} values must be > 0")
    return out


def log_ratio_distance(r_obs: Iterable[float], r_atlas: Iterable[float]) -> float:
    """Euclidean distance in log-ratio space."""
    obs = _as_positive_1d(r_obs, name="r_obs")
    atlas = _as_positive_1d(r_atlas, name="r_atlas")

    if obs.shape != atlas.shape:
        raise ValueError("r_obs and r_atlas must have the same shape")

    diff = np.log(obs) - np.log(atlas)
    return float(np.sqrt(np.sum(diff**2)))


def rank_atlas(r_obs: Iterable[float], atlas_ratios: list[Iterable[float]]) -> list[tuple[int, float]]:
    """Rank atlas candidates by ascending log-ratio distance."""
    scored = [
        (idx, log_ratio_distance(r_obs, r_atlas_i))
        for idx, r_atlas_i in enumerate(atlas_ratios)
    ]
    scored.sort(key=lambda x: x[1])
    return scored


def add_gaussian_noise(value: float, sigma_rel: float, rng_seed: int) -> float:
    """Apply deterministic relative Gaussian noise."""
    if sigma_rel < 0:
        raise ValueError("sigma_rel must be >= 0")
    if not math.isfinite(value):
        raise ValueError("value must be finite")

    rng = np.random.default_rng(rng_seed)
    eps = rng.normal(0.0, sigma_rel)
    return value * (1.0 + eps)
