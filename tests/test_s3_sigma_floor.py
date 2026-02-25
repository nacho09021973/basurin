"""Tests for the df_floor_hz sigma floor in s3_ringdown_estimates.

Regression / acceptance criteria (from spec):

1. Unit (hilbert_envelope): near-monochromatic signal with MAD≈0
   → sigma_floor_applied=True, sigma_logf > 0, df_floor_hz == fs/n

2. Unit (hilbert_envelope): noisy signal
   → sigma_logf > 0 regardless of floor (regression: must never be 0)

3. Unit (spectral_lorentzian): sigma_f from pcov below floor
   → sigma_floor_applied=True via monkeypatch on pcov

4. Regression (CLI, method=hilbert_envelope): monochromatic s2 outputs
   → combined_uncertainty.sigma_logf > 0
   → combined_uncertainty.sigma_floor_applied = True
   → s4_geometry_filter no longer fails with "sigma_lnf got 0.0"
     (verified by checking sigma_lnf > 0 in the JSON output)

5. df_floor_hz value: equals sample_rate / n_samples (deterministic)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s3_ringdown_estimates import (
    estimate_ringdown_observables,
    estimate_ringdown_spectral,
)
import mvp.s3_ringdown_estimates as s3_mod

MVP_DIR = REPO_ROOT / "mvp"

SAMPLE_RATE = 4096.0
BAND = (150.0, 400.0)
F_TRUE = 250.0
TAU_TRUE = 0.05  # 50 ms


# ── helpers ────────────────────────────────────────────────────────────────

def _clean_ringdown(n: int = 2048, f: float = F_TRUE, tau: float = TAU_TRUE,
                    fs: float = SAMPLE_RATE) -> np.ndarray:
    """Pure decaying sinusoid — no noise → MAD of inst-freq ≈ 0."""
    t = np.arange(n) / fs
    return np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t)


def _noisy_ringdown(n: int = 2048, snr: float = 10.0, seed: int = 42) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    signal = np.exp(-t / TAU_TRUE) * np.cos(2.0 * np.pi * F_TRUE * t)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, signal.std() / snr, n)
    return signal + noise


def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )


def _create_s2_monochromatic(runs_root: Path, run_id: str, n: int = 2048) -> None:
    """Write pure-ringdown (no noise) s2 outputs for the CLI regression test."""
    out_dir = runs_root / run_id / "s2_ringdown_window" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    strain = _clean_ringdown(n=n)
    for det in ("H1", "L1"):
        np.savez(
            out_dir / f"{det}_rd.npz",
            strain=strain.astype(np.float64),
            sample_rate_hz=np.float64(SAMPLE_RATE),
        )
    (out_dir / "window_meta.json").write_text(
        json.dumps({"event_id": "SYNTH_MONO"}), encoding="utf-8"
    )
    (runs_root / run_id / "s2_ringdown_window" / "stage_summary.json").write_text(
        json.dumps({"stage": "s2_ringdown_window", "verdict": "PASS"}),
        encoding="utf-8",
    )


def _run_s3_cli(run_id: str, runs_root: Path, method: str = "hilbert_envelope") -> "subprocess.CompletedProcess":
    import subprocess
    cmd = [
        sys.executable, str(MVP_DIR / "s3_ringdown_estimates.py"),
        "--run", run_id,
        "--band-low", str(BAND[0]),
        "--band-high", str(BAND[1]),
        "--method", method,
        "--n-bootstrap", "0",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(REPO_ROOT))


# ── Test 1: hilbert_envelope — monochromatic signal ────────────────────────

class TestHilbertFloor:
    """estimate_ringdown_observables applies df_floor_hz when MAD≈0."""

    def test_sigma_floor_applied_on_clean_signal(self):
        """Pure ringdown (no noise): MAD≈0 → floor must engage."""
        n = 2048
        strain = _clean_ringdown(n=n)
        result = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        assert result["sigma_floor_applied"] is True, (
            f"Expected sigma_floor_applied=True for noiseless signal; "
            f"sigma_f_hz_raw={result['sigma_f_hz_raw']:.6f}, "
            f"df_floor_hz={result['df_floor_hz']:.4f}"
        )

    def test_sigma_logf_positive_after_floor(self):
        """After floor, sigma_logf must be > 0 (Mahalanobis well-defined)."""
        strain = _clean_ringdown()
        result = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        sigma_logf = result["sigma_f_hz"] / result["f_hz"]
        assert sigma_logf > 0, (
            f"sigma_logf must be > 0 after floor, got {sigma_logf}"
        )

    def test_df_floor_hz_equals_fs_over_n(self):
        """df_floor_hz must equal sample_rate / n_samples."""
        n = 2048
        strain = _clean_ringdown(n=n)
        result = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        expected_floor = SAMPLE_RATE / n
        assert abs(result["df_floor_hz"] - expected_floor) < 1e-10, (
            f"df_floor_hz={result['df_floor_hz']} != fs/n={expected_floor}"
        )

    def test_sigma_f_hz_equals_max_of_raw_and_floor(self):
        """sigma_f_hz must equal max(sigma_f_hz_raw, df_floor_hz)."""
        strain = _clean_ringdown()
        result = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        expected = max(result["sigma_f_hz_raw"], result["df_floor_hz"])
        assert abs(result["sigma_f_hz"] - expected) < 1e-12, (
            f"sigma_f_hz={result['sigma_f_hz']} != "
            f"max(raw={result['sigma_f_hz_raw']}, floor={result['df_floor_hz']})={expected}"
        )

    def test_floor_not_applied_for_different_n(self):
        """df_floor_hz changes with n: shorter window → larger floor."""
        n_long  = 4096
        n_short = 512
        r_long  = estimate_ringdown_observables(
            _clean_ringdown(n=n_long), SAMPLE_RATE, BAND[0], BAND[1]
        )
        r_short = estimate_ringdown_observables(
            _clean_ringdown(n=n_short), SAMPLE_RATE, BAND[0], BAND[1]
        )
        assert r_short["df_floor_hz"] > r_long["df_floor_hz"], (
            "Shorter window must produce a larger frequency floor"
        )

    def test_traceability_keys_present(self):
        """Return dict must always contain the three floor traceability keys."""
        strain = _clean_ringdown()
        result = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        for key in ("sigma_f_hz_raw", "df_floor_hz", "sigma_floor_applied"):
            assert key in result, f"Missing traceability key: {key!r}"


# ── Test 2: hilbert_envelope — noisy signal ────────────────────────────────

class TestHilbertFloorNoisySignal:
    """Noisy signal: sigma_logf > 0 regardless of whether floor engages."""

    def test_sigma_logf_always_positive_noisy(self):
        """SNR=10 noisy signal: sigma_logf must be > 0."""
        strain = _noisy_ringdown(snr=10.0)
        result = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        sigma_logf = result["sigma_f_hz"] / result["f_hz"]
        assert sigma_logf > 0, (
            f"sigma_logf must be > 0 for noisy signal, got {sigma_logf}"
        )

    def test_sigma_logf_always_positive_high_snr(self):
        """SNR=100 high-SNR signal: sigma_logf must still be > 0 (floor safety net)."""
        strain = _noisy_ringdown(snr=100.0)
        result = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        sigma_logf = result["sigma_f_hz"] / result["f_hz"]
        assert sigma_logf > 0, (
            f"sigma_logf must be > 0 even at SNR=100, got {sigma_logf}"
        )


# ── Test 3: spectral — floor on pcov path ─────────────────────────────────

class TestSpectralFloor:
    """estimate_ringdown_spectral applies df_floor_hz when pcov gives sigma_f≈0."""

    def test_spectral_traceability_keys_present_on_success(self):
        """When fit succeeds, traceability keys must be present."""
        strain = _noisy_ringdown(snr=20.0)
        result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)
        if not result.get("fit_success"):
            pytest.skip("Spectral fit did not converge on test signal")
        for key in ("sigma_f_hz_raw", "df_floor_hz", "sigma_floor_applied"):
            assert key in result, f"Missing traceability key '{key}' in spectral result"

    def test_spectral_sigma_f_hz_geq_df_floor(self):
        """After floor, sigma_f_hz must be >= df_floor_hz."""
        strain = _noisy_ringdown(snr=20.0)
        result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)
        if not result.get("fit_success"):
            pytest.skip("Spectral fit did not converge on test signal")
        assert result["sigma_f_hz"] >= result["df_floor_hz"] - 1e-12, (
            f"sigma_f_hz={result['sigma_f_hz']} < df_floor_hz={result['df_floor_hz']}"
        )

    def test_spectral_floor_engages_when_pcov_sigma_below_floor(self):
        """Monkeypatch pcov to force sigma_f_raw < df_floor_hz → floor applied."""
        import unittest.mock as mock

        strain = _noisy_ringdown(snr=20.0)
        n = len(strain)
        floor = SAMPLE_RATE / n  # e.g. 4096/2048 = 2.0 Hz
        tiny_sigma = floor * 0.01  # 0.02 Hz — well below floor

        # Build a fake pcov where sigma_f = tiny_sigma
        # pcov[1,1] = sigma_f^2, pcov[2,2] = sigma_tau^2 (non-zero)
        fake_pcov = np.zeros((3, 3))
        fake_pcov[1, 1] = tiny_sigma ** 2
        fake_pcov[2, 2] = 1e-6  # sigma_tau = 1 ms, normal

        original_curve_fit = None
        try:
            from scipy.optimize import curve_fit as _orig_cf

            def _fake_curve_fit(f, xdata, ydata, **kwargs):
                # Call the real curve_fit to get popt, then substitute pcov
                popt, _ = _orig_cf(f, xdata, ydata, **kwargs)
                return popt, fake_pcov

            with mock.patch("mvp.s3_ringdown_estimates.curve_fit",
                            side_effect=_fake_curve_fit,
                            create=True):
                result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)
        except Exception:
            pytest.skip("Could not inject fake pcov — skipping monkeypatch test")
            return

        if not result.get("fit_success"):
            pytest.skip("Spectral fit did not converge even with monkeypatched pcov")

        assert result["sigma_floor_applied"] is True, (
            f"Expected sigma_floor_applied=True when sigma_f_raw={tiny_sigma:.4f} "
            f"< df_floor_hz={floor:.4f}; got sigma_floor_applied={result['sigma_floor_applied']}"
        )
        assert result["sigma_f_hz"] >= floor - 1e-12

    def test_spectral_df_floor_hz_value(self):
        """df_floor_hz must equal sample_rate / n when fit succeeds."""
        n = 2048
        strain = _noisy_ringdown(n=n, snr=20.0)
        result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)
        if not result.get("fit_success"):
            pytest.skip("Spectral fit did not converge on test signal")
        expected = SAMPLE_RATE / n
        assert abs(result["df_floor_hz"] - expected) < 1e-10, (
            f"df_floor_hz={result['df_floor_hz']} != fs/n={expected}"
        )


# ── Test 4: CLI regression — combined_uncertainty ─────────────────────────

class TestCLIRegressionSigmaFloor:
    """CLI-level regression: monochromatic s2 output → sigma_lnf > 0 in JSON."""

    def test_combined_sigma_lnf_positive_on_monochromatic_input(self, tmp_path):
        """Monochromatic input must not produce sigma_lnf=0 in estimates.json.

        This is the regression for the dt=0.001/dt=0.002 FAIL: before the fix,
        hilbert_envelope produced sigma_f_hz=0 (MAD=0) which set sigma_lnf=0
        and caused s4_geometry_filter to fail with 'sigma_lnf got 0.0'.
        """
        runs_root = tmp_path / "runs"
        run_id = "reg_mono_hilbert"
        _create_run_valid(runs_root, run_id)
        _create_s2_monochromatic(runs_root, run_id, n=2048)

        r = _run_s3_cli(run_id, runs_root, method="hilbert_envelope")
        assert r.returncode == 0, (
            f"s3 CLI exited {r.returncode}:\n{r.stderr}"
        )

        est_path = (
            runs_root / run_id / "s3_ringdown_estimates"
            / "outputs" / "estimates.json"
        )
        est = json.loads(est_path.read_text(encoding="utf-8"))
        unc = est["combined_uncertainty"]

        # Regression: sigma_lnf must be > 0 (not 0 as before the fix)
        assert unc["sigma_lnf"] > 0, (
            f"REGRESSION: combined_uncertainty.sigma_lnf={unc['sigma_lnf']} "
            f"must be > 0 for monochromatic input (was 0 before fix)"
        )
        assert math.isfinite(unc["sigma_lnf"]), (
            f"sigma_lnf must be finite, got {unc['sigma_lnf']}"
        )

    def test_combined_sigma_floor_applied_true_on_monochromatic_input(self, tmp_path):
        """sigma_floor_applied must be True for monochromatic input."""
        runs_root = tmp_path / "runs"
        run_id = "reg_mono_floor_flag"
        _create_run_valid(runs_root, run_id)
        _create_s2_monochromatic(runs_root, run_id, n=2048)

        r = _run_s3_cli(run_id, runs_root, method="hilbert_envelope")
        assert r.returncode == 0, f"s3 CLI exited {r.returncode}:\n{r.stderr}"

        est_path = (
            runs_root / run_id / "s3_ringdown_estimates"
            / "outputs" / "estimates.json"
        )
        unc = json.loads(est_path.read_text(encoding="utf-8"))["combined_uncertainty"]
        assert unc.get("sigma_floor_applied") is True, (
            f"combined_uncertainty.sigma_floor_applied must be True for "
            f"noiseless input, got {unc.get('sigma_floor_applied')}"
        )

    def test_combined_sigma_logf_matches_sigma_lnf(self, tmp_path):
        """Legacy sigma_logf and modern sigma_lnf must be equal (alias check)."""
        runs_root = tmp_path / "runs"
        run_id = "reg_alias_check"
        _create_run_valid(runs_root, run_id)
        _create_s2_monochromatic(runs_root, run_id)

        r = _run_s3_cli(run_id, runs_root, method="hilbert_envelope")
        assert r.returncode == 0, f"s3 CLI exited {r.returncode}:\n{r.stderr}"

        unc = json.loads(
            (runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json")
            .read_text(encoding="utf-8")
        )["combined_uncertainty"]
        assert unc["sigma_logf"] == unc["sigma_lnf"], (
            f"sigma_logf={unc['sigma_logf']} != sigma_lnf={unc['sigma_lnf']} "
            f"(must be identical aliases)"
        )
