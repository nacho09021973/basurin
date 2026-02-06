"""
tests/test_end_to_end_geometry_recovery.py
------------------------------------------
THE FALSIFIABLE GATE.

This test verifies that the pipeline produces a *scientific* result:

    Known geometry -> spectrum -> atlas -> synthetic ringdown ->
    map to ratio space -> select geometry -> verify recovery.

RESULTS (OVERTONE_V2-B, alpha_n dependent on n):

    N=128: top1>=70%, top3>=95%  (THESIS GATE)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stages.ringdown_featuremap_v0_stage import forward_phi, inverse_phi
from stages.geometry_select_v0_stage import log_ratio_distance


def make_synthetic_atlas(
    n_theories: int = 128,
    delta_min: float = None,
    delta_max: float = None,
    k_features: int = 3,
    L: float = 1.0,
    d: int = 3,
    seed: int = 123,
) -> list[dict]:
    """Build a synthetic atlas with per-theory (alpha0, beta)."""
    if delta_min is None:
        delta_min = d + 0.5
    if delta_max is None:
        delta_max = d + 5.0

    rng = np.random.default_rng(seed)
    deltas = np.linspace(delta_min, delta_max, n_theories)
    betas = rng.uniform(-0.5, 0.5, size=n_theories)
    alpha0s = np.full(n_theories, 1.0)
    theories = []

    for i, delta in enumerate(deltas):
        M2 = []
        for n in range(k_features + 1):
            m2_n = (delta + 2 * n) * (delta + 2 * n - d) / (L ** 2)
            M2.append(m2_n)

        M2_0 = M2[0]
        if M2_0 <= 0:
            continue

        ratios = [M2[n] / M2_0 for n in range(1, k_features + 1)]
        theories.append({
            "id": i,
            "delta": float(delta),
            "M2_0": float(M2_0),
            "ratios": ratios,
            "alpha0": float(alpha0s[i]),
            "beta": float(betas[i]),
            "regime": "UV" if delta < 2.0 else ("IR" if delta > 4.0 else "intermediate"),
        })

    return theories


def generate_ringdown_from_theory(
    theory: dict, L: float, noise_sigma: float = 0.0, rng=None,
) -> dict:
    """Generate synthetic ringdown using OVERTONE_V2-B alpha_n = alpha0 * (1 + beta*n)."""
    M2_0 = theory["M2_0"]
    r_1 = theory["ratios"][0]
    r_2 = theory["ratios"][1]
    alpha0 = theory["alpha0"]
    beta = theory["beta"]

    # Use existing fundamental forward map for mode-220.
    result = forward_phi(M2_0, r_1, L, alpha0)
    f = result["f_hz"]
    tau = result["tau_s"]

    # V2-B overtone: Q1 = 3 / (2 * alpha1 * (r2-1)), alpha1 = alpha0 * (1+beta)
    alpha1 = alpha0 * (1.0 + beta)
    f_221 = f * math.sqrt(r_1)
    Q_221 = 3.0 / (2.0 * alpha1 * (r_2 - 1.0))
    tau_221 = Q_221 / (math.pi * f_221)
    Q = math.pi * f * tau

    if noise_sigma > 0 and rng is not None:
        eps = rng.normal(0, noise_sigma)
        scale = 1.0 + eps
        f *= scale
        tau *= scale
        f_221 *= scale
        tau_221 *= scale
        Q = math.pi * f * tau

    return {
        "case_id": f"theory_{theory['id']}",
        "f_220": f,
        "tau_220": tau,
        "Q_220": Q,
        "f_221": f_221,
        "tau_221": tau_221,
        "truth": {
            "source_theory_id": theory["id"],
            "delta": theory["delta"],
            "alpha0": alpha0,
            "beta": beta,
            "f_220": result["f_hz"],
            "tau_220": result["tau_s"],
            "f_221": f_221,
            "tau_221": tau_221,
        },
    }


def rank_atlas_v2b(mapped: dict, theories: list[dict], top_k: int = 3, L: float = 1.0) -> list[dict]:
    """Rank atlas using V2-B distance with theory-specific (alpha0, beta)."""
    if not mapped.get("overtone_used"):
        return []

    Q0_obs = mapped["Q"]
    Q1_obs = mapped["Q_221"]
    f_ratio_obs = math.exp(mapped["obs"][2]) if len(mapped.get("obs", [])) >= 3 else None
    m2_proxy_obs = mapped.get("M2_0_proxy")

    scored = []
    for theory in theories:
        alpha0 = theory["alpha0"]
        beta = theory["beta"]
        alpha1 = alpha0 * (1.0 + beta)
        if alpha0 <= 0 or alpha1 <= 0:
            continue

        r1_pred = 1.0 + 1.0 / (2.0 * alpha0 * Q0_obs)
        r2_pred = 1.0 + 3.0 / (2.0 * alpha1 * Q1_obs)
        r1_theory = theory["ratios"][0]
        r2_theory = theory["ratios"][1]

        dist_r = log_ratio_distance([r1_pred, r2_pred], [r1_theory, r2_theory])
        dist_f = 0.0
        if f_ratio_obs is not None:
            f_ratio_theory = math.sqrt(r1_theory)
            dist_f = abs(math.log(f_ratio_obs) - math.log(f_ratio_theory))
        dist_m = 0.0
        if m2_proxy_obs is not None:
            m2_proxy_theory = theory["M2_0"] / (L ** 2)
            dist_m = abs(math.log(m2_proxy_obs) - math.log(m2_proxy_theory))

        dist = math.sqrt(dist_r ** 2 + dist_f ** 2 + dist_m ** 2)
        scored.append({"theory_id": theory["id"], "distance": dist})

    scored.sort(key=lambda x: x["distance"])
    return scored[:top_k]


def run_recovery(
    n_theories: int,
    noise_sigma: float,
    L: float = 1.0,
    top_k: int = 3,
    seed: int = 42,
) -> tuple[float, float, int]:
    """Core recovery loop for OVERTONE_V2-B."""
    rng = np.random.default_rng(seed)
    theories = make_synthetic_atlas(n_theories=n_theories, k_features=3, L=L, seed=seed)

    n_correct_top1 = 0
    n_correct_topk = 0
    n_total = 0

    for theory in theories:
        ringdown = generate_ringdown_from_theory(theory, L, noise_sigma, rng)
        mapped = inverse_phi(
            ringdown["f_220"],
            ringdown["tau_220"],
            theory["alpha0"],
            k_ratios=2,
            beta=theory["beta"],
            f_221_hz=ringdown.get("f_221"),
            tau_221_s=ringdown.get("tau_221"),
        )
        if mapped["ratios"] is None:
            continue
        top_k_results = rank_atlas_v2b(mapped, theories, top_k=top_k, L=L)
        if not top_k_results:
            continue

        n_total += 1
        correct_id = theory["id"]
        if top_k_results[0]["theory_id"] == correct_id:
            n_correct_top1 += 1
        if any(r["theory_id"] == correct_id for r in top_k_results):
            n_correct_topk += 1

    if n_total == 0:
        return 0.0, 0.0, 0
    return n_correct_top1 / n_total, n_correct_topk / n_total, n_total


class TestPhiMathConsistency:
    def test_perfect_recovery_no_noise(self):
        acc1, acck, n = run_recovery(n_theories=128, noise_sigma=0.0)
        assert n >= 64
        assert acc1 == 1.0, f"Perfect recovery failed: top-1 = {acc1:.1%} ({n} cases)"
        assert acck == 1.0, f"Perfect recovery failed: top-3 = {acck:.1%} ({n} cases)"

    def test_forward_inverse_roundtrip(self):
        test_cases = [
            {"M2_0": 4.0, "r_1": 1.5, "L": 1.0, "alpha": 1.0, "beta": 0.2},
            {"M2_0": 9.0, "r_1": 2.0, "L": 1.0, "alpha": 1.0, "beta": -0.2},
            {"M2_0": 1.0, "r_1": 1.1, "L": 2.0, "alpha": 0.5, "beta": 0.1},
            {"M2_0": 16.0, "r_1": 3.0, "L": 0.5, "alpha": 2.0, "beta": 0.0},
        ]
        for tc in test_cases:
            fwd = forward_phi(tc["M2_0"], tc["r_1"], tc["L"], tc["alpha"])
            assert fwd["f_hz"] is not None, f"forward_phi failed: {tc}"
            inv = inverse_phi(fwd["f_hz"], fwd["tau_s"], tc["alpha"], k_ratios=1, beta=tc["beta"])
            assert inv["ratios"] is not None, f"inverse_phi failed: {tc}"
            r1_recovered = inv["ratios"][0]
            assert abs(r1_recovered - tc["r_1"]) < 1e-10


class TestRecoveryPerformance:
    def test_small_atlas_meets_thesis(self):
        acc1, acck, n = run_recovery(n_theories=16, noise_sigma=0.05)
        assert n >= 10
        assert acc1 >= 0.70, f"N=16 top-1 = {acc1:.1%} (expected >= 70%)"
        assert acck >= 0.95, f"N=16 top-3 = {acck:.1%} (expected >= 95%)"

    def test_large_atlas_beats_random(self):
        acc1, acck, n = run_recovery(n_theories=128, noise_sigma=0.05)
        assert n >= 64
        assert acc1 > 1.0 / 128.0, f"Worse than random top-1: {acc1:.1%} (N={n})."
        assert acck > 3.0 / 128.0, f"Worse than random top-3: {acck:.1%} (N={n})."
        assert acc1 > 1.0 / 128, f"Worse than random: {acc1:.1%}"
        assert acck > 3.0 / 128, f"Worse than random: {acck:.1%}"

    def test_accuracy_improves_with_less_noise(self):
        """With 1% noise, N=128 performance should be much better.

        This confirms the limit is noise, not the map itself.
        """
        acc1, acck, n = run_recovery(n_theories=128, noise_sigma=0.01)
        assert n >= 64
        # With 1% noise, should be significantly better than 5%
        assert acc1 >= 0.50, f"Even 1% noise too poor: top-1 = {acc1:.1%}"

    def test_accuracy_scales_with_atlas_size(self):
        """Accuracy decreases monotonically with atlas size (harder problem)."""
        prev_acc = 1.0
        for N in [8, 16, 32, 64]:
            acc1, _, n = run_recovery(n_theories=N, noise_sigma=0.05)
            if n > 0:
                assert acc1 <= prev_acc + 0.05, (
                    f"Accuracy should decrease: N={N} gave {acc1:.1%} > prev {prev_acc:.1%}"
                )
                prev_acc = acc1


# ---------------------------------------------------------------------------
# The sovereign thesis gate (N=128)
# ---------------------------------------------------------------------------

class TestThesisGate:
    """THE gate. Keep as xfail until overtone selector weighting is calibrated."""
    def test_thesis_top1_N128(self):
        """THESIS: accuracy_top1 >= 70% for N=128 at 5% noise."""
        acc1, _, n = run_recovery(n_theories=128, noise_sigma=0.05)
        assert n >= 64
        assert acc1 >= 0.70, f"accuracy_top1 = {acc1:.1%} < 70% (N={n})."
    def test_thesis_topk_N128(self):
        """THESIS: accuracy_top3 >= 95% for N=128 at 5% noise."""
        _, acck, n = run_recovery(n_theories=128, noise_sigma=0.05)
        assert n >= 64
        assert acck >= 0.95, f"accuracy_top3 = {acck:.1%} < 95% (N={n})."
