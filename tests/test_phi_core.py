import math

import numpy as np
import pytest

from phi_core import (
    add_gaussian_noise,
    forward_model,
    inverse_model,
    rank_atlas,
)


def test_forward_q_consistency_exact():
    out = forward_model(M2_0=4.0, r1=1.25, L=2.0, alpha=1.0)
    assert np.isclose(out["Q"], math.pi * out["f"] * out["tau"], atol=1e-15, rtol=0.0)


def test_roundtrip_no_noise():
    M2_0 = 4.0
    L = 1.0
    r1 = 1.23456789
    alpha = 1.0

    out_fwd = forward_model(M2_0=M2_0, r1=r1, L=L, alpha=alpha)
    out_inv = inverse_model(out_fwd["f"], out_fwd["tau"], alpha=alpha)

    assert abs(out_inv["r1_pred"] - r1) < 1e-10


def test_ranking_perfect_without_noise_n16():
    n = 16
    r1_list = np.linspace(1.05, 1.50, n)
    atlas_ratios = [np.array([r]) for r in r1_list]

    for i in range(n):
        r_obs = np.array([r1_list[i]])
        ranking = rank_atlas(r_obs, atlas_ratios)
        assert ranking[0][0] == i


def test_degradation_with_noise_n16_sigma5pct():
    n = 16
    r1_list = np.linspace(1.05, 1.50, n)
    atlas_ratios = [np.array([r]) for r in r1_list]

    M2_0 = 4.0
    L = 1.0
    alpha = 1.0

    correct = 0
    for i in range(n):
        out = forward_model(M2_0=M2_0, r1=r1_list[i], L=L, alpha=alpha)
        f_noisy = add_gaussian_noise(out["f"], sigma_rel=0.05, rng_seed=12345 + 2 * i)
        tau_noisy = add_gaussian_noise(out["tau"], sigma_rel=0.05, rng_seed=12345 + 2 * i + 1)
        out_inv = inverse_model(f_noisy, tau_noisy, alpha=alpha)
        r_obs = np.array([out_inv["r1_pred"]])

        ranking = rank_atlas(r_obs, atlas_ratios)
        if ranking[0][0] == i:
            correct += 1

    accuracy_top1 = correct / n
    assert accuracy_top1 >= 0.65


@pytest.mark.parametrize(
    "kwargs",
    [
        {"M2_0": 4.0, "r1": 1.0, "L": 1.0, "alpha": 1.0},
        {"M2_0": 4.0, "r1": 0.99, "L": 1.0, "alpha": 1.0},
        {"M2_0": 4.0, "r1": 1.2, "L": 1.0, "alpha": 0.0},
        {"M2_0": 4.0, "r1": 1.2, "L": 1.0, "alpha": -0.1},
        {"M2_0": 4.0, "r1": 1.2, "L": 0.0, "alpha": 1.0},
        {"M2_0": 4.0, "r1": 1.2, "L": -1.0, "alpha": 1.0},
        {"M2_0": 0.0, "r1": 1.2, "L": 1.0, "alpha": 1.0},
        {"M2_0": -1.0, "r1": 1.2, "L": 1.0, "alpha": 1.0},
    ],
)
def test_forward_validation_errors(kwargs):
    with pytest.raises(ValueError):
        forward_model(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"f_obs": 0.0, "tau_obs": 1.0, "alpha": 1.0},
        {"f_obs": -1.0, "tau_obs": 1.0, "alpha": 1.0},
        {"f_obs": 1.0, "tau_obs": 0.0, "alpha": 1.0},
        {"f_obs": 1.0, "tau_obs": -1.0, "alpha": 1.0},
        {"f_obs": 1.0, "tau_obs": 1.0, "alpha": 0.0},
        {"f_obs": 1.0, "tau_obs": 1.0, "alpha": -1.0},
    ],
)
def test_inverse_validation_errors(kwargs):
    with pytest.raises(ValueError):
        inverse_model(**kwargs)
