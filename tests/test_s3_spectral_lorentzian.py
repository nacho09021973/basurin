"""Tests for the spectral_lorentzian estimator in s3_ringdown_estimates.

Test 1 — Accuracy on synthetic signal (SNR=20):
    |f_est - f_true| / f_true < 0.02  (< 2%)
    |tau_est - tau_true| / tau_true < 0.05  (< 5%)

Test 2 — Bias comparison vs hilbert_envelope:
    hilbert_envelope error > 10% on f or Q (documents known bias)
    spectral_lorentzian error < 5%

Test 3 — Fallback when curve_fit fails:
    Pure noise / no usable PSD bins → no exception, fit_success=False

Test 4 — Determinism:
    Two calls with identical inputs → bit-identical outputs
    CLI-level: two subprocess runs produce same SHA256 on estimates.json

Test 5 — Contract keys:
    combined_uncertainty has all keys required by the downstream contract
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s3_ringdown_estimates import (
    estimate_ringdown_observables,
    estimate_ringdown_spectral,
)


# ── Helpers ───────────────────────────────────────────────────────────────

F_TRUE = 250.0       # Hz
TAU_TRUE = 0.05      # s  (50 ms, typical BBH ringdown)
SAMPLE_RATE = 4096.0
DURATION = 0.5       # s
SNR = 20.0
BAND = (150.0, 400.0)


def _make_ringdown(
    f_true: float = F_TRUE,
    tau_true: float = TAU_TRUE,
    sample_rate: float = SAMPLE_RATE,
    duration: float = DURATION,
    snr: float = SNR,
    seed: int = 42,
) -> np.ndarray:
    """h(t) = A * exp(-t/tau) * cos(2*pi*f*t) + noise at given SNR."""
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    signal = np.exp(-t / tau_true) * np.cos(2.0 * np.pi * f_true * t)
    rng = np.random.default_rng(seed)
    noise_sigma = signal.std() / snr
    noise = rng.normal(0.0, noise_sigma, n)
    return signal + noise


def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )


def _create_s2_outputs(
    runs_root: Path,
    run_id: str,
    *,
    seed: int = 42,
    duration: float = 0.5,
) -> None:
    """Write synthetic s2 outputs that s3 CLI can consume."""
    out_dir = runs_root / run_id / "s2_ringdown_window" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for det_seed_offset, det in enumerate(("H1", "L1")):
        strain = _make_ringdown(seed=seed + det_seed_offset, duration=duration)
        np.savez(
            out_dir / f"{det}_rd.npz",
            strain=strain.astype(np.float64),
            sample_rate_hz=np.float64(SAMPLE_RATE),
        )

    (out_dir / "window_meta.json").write_text(
        json.dumps({"event_id": "SYNTHETIC_EVENT"}), encoding="utf-8"
    )
    stage_dir = runs_root / run_id / "s2_ringdown_window"
    (stage_dir / "stage_summary.json").write_text(
        json.dumps({"stage": "s2_ringdown_window", "verdict": "PASS"}),
        encoding="utf-8",
    )


def _run_s3(
    run_id: str,
    runs_root: Path,
    method: str = "spectral_lorentzian",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(MVP_DIR / "s3_ringdown_estimates.py"),
        "--run", run_id,
        "--band-low", str(BAND[0]),
        "--band-high", str(BAND[1]),
        "--method", method,
        "--n-bootstrap", "0",
    ]
    if extra_args:
        cmd += extra_args
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(REPO_ROOT))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Test 1: Accuracy on synthetic signal ─────────────────────────────────

class TestSpectralAccuracy:
    """spectral_lorentzian achieves < 2% error on f and < 5% error on tau."""

    def test_frequency_error_below_2pct(self):
        strain = _make_ringdown()
        result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)

        assert result["fit_success"], (
            f"spectral fit failed on clean synthetic signal: {result}"
        )
        err_f = abs(result["f_hz"] - F_TRUE) / F_TRUE
        assert err_f < 0.02, (
            f"Frequency error {err_f:.3%} >= 2% "
            f"(f_est={result['f_hz']:.2f}, f_true={F_TRUE})"
        )

    def test_tau_error_below_5pct(self):
        strain = _make_ringdown()
        result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)

        assert result["fit_success"], (
            f"spectral fit failed on clean synthetic signal: {result}"
        )
        err_tau = abs(result["tau_s"] - TAU_TRUE) / TAU_TRUE
        assert err_tau < 0.05, (
            f"Tau error {err_tau:.3%} >= 5% "
            f"(tau_est={result['tau_s']:.4f}, tau_true={TAU_TRUE})"
        )

    def test_Q_relation_holds(self):
        """Q must equal pi * f * tau (internal consistency)."""
        strain = _make_ringdown()
        result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)

        if not result["fit_success"]:
            pytest.skip("fit did not converge — Q relation not testable")

        expected_Q = math.pi * result["f_hz"] * result["tau_s"]
        assert abs(result["Q"] - expected_Q) < 1e-9, (
            f"Q={result['Q']:.4f} != pi*f*tau={expected_Q:.4f}"
        )

    def test_uncertainties_positive_and_finite(self):
        """sigma fields must be positive and finite on a clean signal."""
        strain = _make_ringdown()
        result = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)

        if not result["fit_success"]:
            pytest.skip("fit did not converge — sigma fields not testable")

        for key in ("sigma_f_hz", "sigma_tau_s", "sigma_Q"):
            val = result[key]
            assert math.isfinite(val), f"{key} not finite: {val}"
            assert val > 0.0, f"{key} must be > 0, got {val}"


# ── Test 2: Bias comparison hilbert vs spectral ───────────────────────────

class TestBiasComparison:
    """spectral_lorentzian corrects the known hilbert_envelope bias."""

    def _errors(self, seed: int = 42):
        strain = _make_ringdown(seed=seed)

        hilbert = estimate_ringdown_observables(
            strain, SAMPLE_RATE, BAND[0], BAND[1]
        )
        spectral = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)

        Q_true = math.pi * F_TRUE * TAU_TRUE

        err_f_h  = abs(hilbert["f_hz"] - F_TRUE) / F_TRUE
        err_Q_h  = abs(hilbert["Q"] - Q_true) / Q_true

        err_f_s  = abs(spectral["f_hz"] - F_TRUE) / F_TRUE if spectral["fit_success"] else None
        err_Q_s  = abs(spectral["Q"] - Q_true) / Q_true if spectral["fit_success"] else None

        return err_f_h, err_Q_h, err_f_s, err_Q_s

    def test_hilbert_has_significant_bias(self):
        """hilbert_envelope must show > 10% error on f or Q."""
        err_f_h, err_Q_h, _, _ = self._errors()
        has_bias = (err_f_h > 0.10) or (err_Q_h > 0.10)
        assert has_bias, (
            f"hilbert_envelope did not show expected bias: "
            f"err_f={err_f_h:.2%}, err_Q={err_Q_h:.2%}"
        )

    def test_spectral_has_low_bias(self):
        """spectral_lorentzian must achieve < 5% error on both f and Q."""
        _, _, err_f_s, err_Q_s = self._errors()

        if err_f_s is None:
            pytest.fail("spectral_lorentzian fit failed on synthetic signal with SNR=20")

        assert err_f_s < 0.05, (
            f"spectral f error {err_f_s:.2%} >= 5%"
        )
        assert err_Q_s < 0.05, (
            f"spectral Q error {err_Q_s:.2%} >= 5%"
        )

    def test_spectral_strictly_better_than_hilbert(self):
        """spectral error < hilbert error for both f and Q."""
        err_f_h, err_Q_h, err_f_s, err_Q_s = self._errors()

        if err_f_s is None:
            pytest.skip("spectral fit failed — comparison skipped")

        assert err_f_s < err_f_h, (
            f"spectral f error ({err_f_s:.2%}) not better than "
            f"hilbert ({err_f_h:.2%})"
        )
        assert err_Q_s < err_Q_h, (
            f"spectral Q error ({err_Q_s:.2%}) not better than "
            f"hilbert ({err_Q_h:.2%})"
        )


# ── Test 3: Fallback when curve_fit fails ────────────────────────────────

class TestFallback:
    """estimate_ringdown_spectral handles pathological inputs without raising."""

    def test_no_exception_on_pure_noise(self):
        """Large pure-noise array must not raise any exception."""
        rng = np.random.default_rng(99)
        noise = rng.normal(0.0, 1.0, 4096)
        result = estimate_ringdown_spectral(noise, SAMPLE_RATE, BAND)
        # Must return a dict with all required keys, no exception
        assert isinstance(result, dict)
        for key in ("f_hz", "tau_s", "Q", "sigma_f_hz", "sigma_tau_s",
                    "sigma_Q", "cov_logf_logQ", "fit_success", "fit_residual"):
            assert key in result, f"Missing key '{key}' in fallback result"

    def test_fit_success_false_insufficient_psd_bins(self):
        """Extremely narrow band → 0 PSD bins → fit_success=False."""
        rng = np.random.default_rng(99)
        noise = rng.normal(0.0, 1.0, 4096)
        # With nfft = max(4*4096, 4096*4) = 16384, bin spacing = 0.25 Hz.
        # Band [250.0, 250.1] spans 0.1 Hz → 0 bins fall in it → fit_success=False.
        result = estimate_ringdown_spectral(noise, SAMPLE_RATE, (250.0, 250.1))
        assert result["fit_success"] is False, (
            f"Expected fit_success=False with 0.1 Hz band, got {result['fit_success']}"
        )

    def test_no_exception_on_too_short_array(self):
        """Array shorter than minimum (< 16 samples) must not raise."""
        short = np.ones(8)
        result = estimate_ringdown_spectral(short, SAMPLE_RATE, BAND)
        assert isinstance(result, dict)
        assert result["fit_success"] is False

    def test_cli_no_crash_on_noise_input(self, tmp_path):
        """CLI: pure noise NPZ must not crash (no Python traceback)."""
        runs_root = tmp_path / "runs"
        run_id = "test_fallback_noise"
        _create_run_valid(runs_root, run_id)

        out_dir = runs_root / run_id / "s2_ringdown_window" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(99)
        noise = rng.normal(0.0, 1.0, 4096)
        np.savez(out_dir / "H1_rd.npz",
                 strain=noise.astype(np.float64),
                 sample_rate_hz=np.float64(SAMPLE_RATE))
        (out_dir / "window_meta.json").write_text(
            json.dumps({"event_id": "NOISE_EVENT"}), encoding="utf-8"
        )
        (runs_root / run_id / "s2_ringdown_window" / "stage_summary.json").write_text(
            json.dumps({"stage": "s2_ringdown_window", "verdict": "PASS"}),
            encoding="utf-8",
        )

        r = _run_s3(run_id, runs_root)
        # Must not crash with a Python traceback regardless of exit code
        assert "Traceback" not in r.stderr, (
            f"Unexpected Python traceback in stderr:\n{r.stderr}"
        )

    def test_fit_success_false_unit_level_narrow_band(self):
        """Unit test: very narrow band gives < 5 PSD bins → fit_success=False."""
        rng = np.random.default_rng(99)
        # With nfft >= sample_rate*4 = 16384, bin spacing = 0.25 Hz.
        # Band [250.0, 250.1] Hz contains 0 bins (none fall in 0.25 Hz grid within 0.1 Hz).
        noise = rng.normal(0.0, 1.0, 4096)
        result = estimate_ringdown_spectral(noise, SAMPLE_RATE, (250.0, 250.1))
        assert result["fit_success"] is False, (
            f"Expected fit_success=False for 0.1 Hz band, got True"
        )


# ── Test 4: Determinism ──────────────────────────────────────────────────

class TestDeterminism:
    """Same inputs → bit-identical outputs."""

    def test_function_level_determinism(self):
        """Calling estimate_ringdown_spectral twice gives identical floats."""
        strain = _make_ringdown()
        r1 = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)
        r2 = estimate_ringdown_spectral(strain, SAMPLE_RATE, BAND)

        for key in ("f_hz", "tau_s", "Q", "sigma_f_hz", "sigma_tau_s",
                    "sigma_Q", "cov_logf_logQ", "fit_success", "fit_residual"):
            assert r1[key] == r2[key], (
                f"Non-deterministic key '{key}': {r1[key]} vs {r2[key]}"
            )

    def test_cli_estimates_json_sha256_identical(self, tmp_path):
        """Two CLI runs with identical inputs produce the same SHA256."""
        runs_root = tmp_path / "runs"
        hashes = []

        for i in range(2):
            run_id = f"det_spectral_{i}"
            _create_run_valid(runs_root, run_id)
            _create_s2_outputs(runs_root, run_id, seed=42)

            r = _run_s3(run_id, runs_root)
            assert r.returncode == 0, (
                f"s3 run {i} failed (rc={r.returncode}):\n{r.stderr}"
            )

            est_path = (
                runs_root / run_id / "s3_ringdown_estimates"
                / "outputs" / "estimates.json"
            )
            hashes.append(_sha256_file(est_path))

        assert hashes[0] == hashes[1], (
            f"Non-deterministic SHA256: run0={hashes[0]}, run1={hashes[1]}"
        )


# ── Test 5: Contract keys ────────────────────────────────────────────────

class TestContractKeys:
    """estimates.json must satisfy the downstream key contract."""

    # Required keys in combined_uncertainty (downstream s4 reads these)
    REQUIRED_COMBINED_UNCERTAINTY_KEYS = {
        "sigma_f_hz", "sigma_tau_s", "sigma_Q",
        "cov_logf_logQ",
        "sigma_logf", "sigma_logQ",
        "sigma_lnf", "sigma_lnQ",
        "r",
    }

    # Required keys in combined
    REQUIRED_COMBINED_KEYS = {"f_hz", "tau_s", "Q"}

    def _run_and_load(self, tmp_path: Path, run_id: str = "test_contract") -> dict:
        runs_root = tmp_path / "runs"
        _create_run_valid(runs_root, run_id)
        _create_s2_outputs(runs_root, run_id)

        r = _run_s3(run_id, runs_root)
        assert r.returncode == 0, f"s3 failed (rc={r.returncode}):\n{r.stderr}"

        est_path = (
            runs_root / run_id / "s3_ringdown_estimates"
            / "outputs" / "estimates.json"
        )
        return json.loads(est_path.read_text(encoding="utf-8"))

    def test_combined_has_required_keys(self, tmp_path):
        est = self._run_and_load(tmp_path, "test_contract_combined")
        combined = est.get("combined", {})
        for key in self.REQUIRED_COMBINED_KEYS:
            assert key in combined, f"Missing combined.{key}"

    def test_combined_uncertainty_has_required_keys(self, tmp_path):
        est = self._run_and_load(tmp_path, "test_contract_unc")
        unc = est.get("combined_uncertainty", {})
        for key in self.REQUIRED_COMBINED_UNCERTAINTY_KEYS:
            assert key in unc, f"Missing combined_uncertainty.{key}"

    def test_method_field_is_spectral_lorentzian(self, tmp_path):
        """Default CLI run must record method='spectral_lorentzian'."""
        est = self._run_and_load(tmp_path, "test_contract_method")
        assert est.get("method") == "spectral_lorentzian", (
            f"Expected method='spectral_lorentzian', got '{est.get('method')}'"
        )

    def test_method_field_hilbert_when_requested(self, tmp_path):
        """--method hilbert_envelope must record that method."""
        runs_root = tmp_path / "runs"
        run_id = "test_hilbert_method"
        _create_run_valid(runs_root, run_id)
        _create_s2_outputs(runs_root, run_id)

        r = _run_s3(run_id, runs_root, method="hilbert_envelope")
        assert r.returncode == 0, f"s3 failed:\n{r.stderr}"

        est_path = (
            runs_root / run_id / "s3_ringdown_estimates"
            / "outputs" / "estimates.json"
        )
        est = json.loads(est_path.read_text(encoding="utf-8"))
        assert est.get("method") == "hilbert_envelope"

    def test_sigma_lnf_sigma_lnQ_finite_and_positive(self, tmp_path):
        """Modern canonical keys sigma_lnf and sigma_lnQ must be > 0 and finite."""
        est = self._run_and_load(tmp_path, "test_contract_ln")
        unc = est["combined_uncertainty"]
        for key in ("sigma_lnf", "sigma_lnQ"):
            val = unc[key]
            assert math.isfinite(val), f"{key} not finite: {val}"
            assert val > 0.0, f"{key} must be > 0, got {val}"

    def test_r_in_open_interval(self, tmp_path):
        """Correlation r must satisfy |r| < 1."""
        est = self._run_and_load(tmp_path, "test_contract_r")
        r_val = est["combined_uncertainty"]["r"]
        assert math.isfinite(r_val), f"r not finite: {r_val}"
        assert abs(r_val) < 1.0, f"|r|={abs(r_val):.6f} >= 1"

    def test_q_equals_pi_f_tau(self, tmp_path):
        """Q == pi * f * tau must hold in combined."""
        est = self._run_and_load(tmp_path, "test_contract_q")
        c = est["combined"]
        expected = math.pi * c["f_hz"] * c["tau_s"]
        assert abs(c["Q"] - expected) < 1e-6, (
            f"Q={c['Q']:.6f} != pi*f*tau={expected:.6f}"
        )
