#!/usr/bin/env python3
"""Validate delta_BIC = -10 threshold against synthetic injections.

Purpose:
    Verify that compute_model_comparison correctly discriminates between
    1-mode (220-only) and 2-mode (220+221) synthetic ringdown signals.
    Without this validation, the threshold is an unanchored number.

Scenarios:
    1. 1-mode signal, high SNR  → delta_BIC should NOT prefer 2-mode
    2. 1-mode signal, low SNR   → delta_BIC should NOT prefer 2-mode
    3. 2-mode signal, high SNR  → delta_BIC SHOULD prefer 2-mode
    4. 2-mode signal, low SNR   → delta_BIC may or may not cross (boundary check)

Methodology:
    - Use oracle (true) parameters for mode_220 and mode_221 dicts to isolate
      BIC behavior from estimation error.
    - Use colored noise from experiment_injection_suite infrastructure.
    - Run multiple seeds per scenario for statistical robustness.

Reference:
    Kass & Raftery (1995): |ΔBIC| > 10 = "very strong" evidence.
    Threshold -10 means BIC_2mode is at least 10 lower than BIC_1mode.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# --- path setup (same pattern as other tests) ---
_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from mvp.experiment_injection_suite import (
    FS,
    _generate_colored_noise,
)
from mvp.s3b_multimode_estimates import compute_model_comparison

# ---------------------------------------------------------------------------
# Physical parameters (GW150914-like ringdown)
# ---------------------------------------------------------------------------
F_220 = 251.0    # Hz — fundamental QNM frequency (l=m=2, n=0)
Q_220 = 4.0      # quality factor
TAU_220 = Q_220 / (math.pi * F_220)

# Mode 221: overtone — higher frequency, lower Q (shorter-lived)
F_221 = 274.0    # Hz — first overtone
Q_221 = 2.0      # quality factor (overtones decay faster)
TAU_221 = Q_221 / (math.pi * F_221)

DURATION = 0.5   # seconds
N_SAMPLES = int(DURATION * FS)
THRESHOLD = -10.0
N_SEEDS = 20     # number of noise realizations per scenario


def _make_mode_dict(f_hz: float, Q: float, label: str) -> dict:
    """Build mode dict in the format expected by compute_model_comparison."""
    return {
        "ln_f": math.log(f_hz),
        "ln_Q": math.log(Q),
        "label": label,
    }


def _make_signal_1mode(snr: float, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic 1-mode (220-only) ringdown + colored noise."""
    t = np.arange(N_SAMPLES) / FS
    signal = np.exp(-t / TAU_220) * np.cos(2.0 * math.pi * F_220 * t)

    noise = _generate_colored_noise(N_SAMPLES, FS, rng)
    noise_std = float(np.std(noise)) + 1e-30
    noise = noise / noise_std

    signal_rms = float(np.sqrt(np.mean(signal ** 2))) + 1e-30
    A = snr / signal_rms

    return A * signal + noise


def _make_signal_2mode(
    snr_220: float, snr_221: float, rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic 2-mode (220+221) ringdown + colored noise."""
    t = np.arange(N_SAMPLES) / FS

    sig_220 = np.exp(-t / TAU_220) * np.cos(2.0 * math.pi * F_220 * t)
    sig_221 = np.exp(-t / TAU_221) * np.cos(2.0 * math.pi * F_221 * t)

    noise = _generate_colored_noise(N_SAMPLES, FS, rng)
    noise_std = float(np.std(noise)) + 1e-30
    noise = noise / noise_std

    rms_220 = float(np.sqrt(np.mean(sig_220 ** 2))) + 1e-30
    rms_221 = float(np.sqrt(np.mean(sig_221 ** 2))) + 1e-30

    A_220 = snr_220 / rms_220
    A_221 = snr_221 / rms_221

    return A_220 * sig_220 + A_221 * sig_221 + noise


def _run_scenario(signal: np.ndarray) -> dict:
    """Run compute_model_comparison with oracle parameters."""
    mode_220 = _make_mode_dict(F_220, Q_220, "220")
    mode_221 = _make_mode_dict(F_221, Q_221, "221")

    return compute_model_comparison(
        signal=signal,
        fs=FS,
        mode_220=mode_220,
        mode_221=mode_221,
        ok_220=True,
        ok_221=True,
        delta_bic_threshold=THRESHOLD,
    )


# ===================================================================
# Scenario 1: 1-mode signal, high SNR → should NOT prefer 2-mode
# ===================================================================
class TestOneMode:
    """1-mode signal: delta_BIC should not cross -10 (no real 221 present)."""

    def test_high_snr_rejects_two_mode(self):
        """SNR=30: strong 220-only signal. BIC should penalize extra parameters."""
        n_preferred_2mode = 0
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed=1000 + seed)
            signal = _make_signal_1mode(snr=30.0, rng=rng)
            result = _run_scenario(signal)

            assert result["valid_bic_1mode"] is True
            assert result["valid_bic_2mode"] is True
            assert result["delta_bic"] is not None

            if result["decision"]["two_mode_preferred"]:
                n_preferred_2mode += 1

        # With no real 221, false positive rate should be very low
        false_positive_rate = n_preferred_2mode / N_SEEDS
        assert false_positive_rate <= 0.10, (
            f"False positive rate {false_positive_rate:.0%} exceeds 10% "
            f"for 1-mode signal at SNR=30 ({n_preferred_2mode}/{N_SEEDS} seeds)"
        )

    def test_low_snr_rejects_two_mode(self):
        """SNR=8: weak 220-only signal. Should still not prefer 2-mode."""
        n_preferred_2mode = 0
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed=2000 + seed)
            signal = _make_signal_1mode(snr=8.0, rng=rng)
            result = _run_scenario(signal)

            if result["delta_bic"] is not None and result["decision"]["two_mode_preferred"]:
                n_preferred_2mode += 1

        false_positive_rate = n_preferred_2mode / N_SEEDS
        assert false_positive_rate <= 0.15, (
            f"False positive rate {false_positive_rate:.0%} exceeds 15% "
            f"for 1-mode signal at SNR=8 ({n_preferred_2mode}/{N_SEEDS} seeds)"
        )


# ===================================================================
# Scenario 2: 2-mode signal, high SNR → SHOULD prefer 2-mode
# ===================================================================
class TestTwoMode:
    """2-mode signal: delta_BIC should cross -10 when 221 is genuinely present."""

    def test_high_snr_both_modes_prefers_two_mode(self):
        """SNR_220=30, SNR_221=15: clear 2-mode signal."""
        n_preferred_2mode = 0
        delta_bics = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed=3000 + seed)
            signal = _make_signal_2mode(snr_220=30.0, snr_221=15.0, rng=rng)
            result = _run_scenario(signal)

            assert result["valid_bic_1mode"] is True
            assert result["valid_bic_2mode"] is True
            assert result["delta_bic"] is not None

            delta_bics.append(result["delta_bic"])
            if result["decision"]["two_mode_preferred"]:
                n_preferred_2mode += 1

        detection_rate = n_preferred_2mode / N_SEEDS
        median_delta_bic = float(np.median(delta_bics))

        assert detection_rate >= 0.80, (
            f"Detection rate {detection_rate:.0%} below 80% "
            f"for 2-mode signal at SNR_220=30, SNR_221=15 "
            f"({n_preferred_2mode}/{N_SEEDS} seeds, "
            f"median ΔBIC={median_delta_bic:.1f})"
        )
        assert median_delta_bic < THRESHOLD, (
            f"Median ΔBIC={median_delta_bic:.1f} not below threshold {THRESHOLD} "
            f"for 2-mode signal at high SNR"
        )

    def test_moderate_snr_221_detects(self):
        """SNR_220=20, SNR_221=10: moderate 221 signal."""
        n_preferred_2mode = 0
        delta_bics = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed=4000 + seed)
            signal = _make_signal_2mode(snr_220=20.0, snr_221=10.0, rng=rng)
            result = _run_scenario(signal)

            if result["delta_bic"] is not None:
                delta_bics.append(result["delta_bic"])
                if result["decision"]["two_mode_preferred"]:
                    n_preferred_2mode += 1

        detection_rate = n_preferred_2mode / N_SEEDS
        median_delta_bic = float(np.median(delta_bics)) if delta_bics else 0.0

        # At moderate SNR we expect decent detection but allow some misses
        assert detection_rate >= 0.60, (
            f"Detection rate {detection_rate:.0%} below 60% "
            f"for 2-mode signal at SNR_220=20, SNR_221=10 "
            f"({n_preferred_2mode}/{N_SEEDS} seeds, "
            f"median ΔBIC={median_delta_bic:.1f})"
        )


# ===================================================================
# Scenario 3: Boundary — weak 221, should be ambiguous
# ===================================================================
class TestBoundary:
    """Weak 221 signal: characterize threshold behavior without hard pass/fail."""

    def test_weak_221_characterization(self):
        """SNR_220=15, SNR_221=3: very weak overtone. Report behavior."""
        n_preferred_2mode = 0
        delta_bics = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed=5000 + seed)
            signal = _make_signal_2mode(snr_220=15.0, snr_221=3.0, rng=rng)
            result = _run_scenario(signal)

            if result["delta_bic"] is not None:
                delta_bics.append(result["delta_bic"])
                if result["decision"]["two_mode_preferred"]:
                    n_preferred_2mode += 1

        detection_rate = n_preferred_2mode / N_SEEDS
        median_delta_bic = float(np.median(delta_bics)) if delta_bics else 0.0
        min_delta_bic = float(np.min(delta_bics)) if delta_bics else 0.0
        max_delta_bic = float(np.max(delta_bics)) if delta_bics else 0.0

        # This is characterization, not a hard gate. We just verify it runs
        # and record the behavior for human review.
        print(
            f"\n[BOUNDARY] SNR_220=15, SNR_221=3:\n"
            f"  detection_rate = {detection_rate:.0%} ({n_preferred_2mode}/{N_SEEDS})\n"
            f"  median ΔBIC = {median_delta_bic:.1f}\n"
            f"  range ΔBIC = [{min_delta_bic:.1f}, {max_delta_bic:.1f}]\n"
        )

        # Sanity: the test produced valid results
        assert len(delta_bics) == N_SEEDS


# ===================================================================
# Scenario 4: Pure noise — no signal at all
# ===================================================================
class TestPureNoise:
    """Pure noise: should never prefer 2-mode model."""

    def test_noise_only_rejects_both(self):
        """No signal, just colored noise. BIC should not prefer extra modes."""
        n_preferred_2mode = 0
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed=6000 + seed)
            noise = _generate_colored_noise(N_SAMPLES, FS, rng)
            result = _run_scenario(noise)

            if result["delta_bic"] is not None and result["decision"]["two_mode_preferred"]:
                n_preferred_2mode += 1

        false_positive_rate = n_preferred_2mode / N_SEEDS
        assert false_positive_rate <= 0.10, (
            f"False positive rate {false_positive_rate:.0%} exceeds 10% "
            f"on pure noise ({n_preferred_2mode}/{N_SEEDS} seeds)"
        )
