#!/usr/bin/env python3
"""MVP Stage 3: Estimate ringdown observables (f, tau, Q) per event.

CLI:
    python mvp/s3_ringdown_estimates.py --run <run_id> \
        [--band-low 150] [--band-high 400] \
        [--method {hilbert_envelope,spectral_lorentzian}]

Inputs:  runs/<run>/s2_ringdown_window/outputs/{H1,L1}_rd.npz
Outputs: runs/<run>/s3_ringdown_estimates/outputs/estimates.json

Method (default): spectral_lorentzian — PSD Lorentzian fit (unbiased).
Legacy method:    hilbert_envelope  — analytic signal phase/envelope.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, check_inputs, finalize, abort
from basurin_io import write_json_atomic

STAGE = "s3_ringdown_estimates"


def estimate_ringdown_observables(
    strain: np.ndarray, fs: float,
    band_low: float = 150.0, band_high: float = 400.0,
) -> dict[str, Any]:
    """Estimate f, tau, Q from a ringdown strain segment via Hilbert envelope.

    Returns point estimates and uncertainty fields:
      f_hz, tau_s, Q, snr_peak          – point estimates (unchanged)
      sigma_f_hz, sigma_tau_s, sigma_Q   – 1-sigma uncertainties
      cov_logf_logQ                      – covariance in (ln f, ln Q) space
    """
    from scipy.signal import butter, sosfilt, hilbert

    n = strain.size
    if n < 16:
        raise ValueError(f"Strain too short: {n} samples")

    nyquist = fs / 2.0
    if band_high >= nyquist:
        band_high = nyquist * 0.95
    if band_low >= band_high:
        raise ValueError(f"Invalid band: [{band_low}, {band_high}] Hz")

    sos = butter(4, [band_low, band_high], btype="band", fs=fs, output="sos")
    filtered = sosfilt(sos, strain)

    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))

    inst_freq = np.diff(inst_phase) * fs / (2.0 * np.pi)
    valid_mask = (inst_freq > band_low * 0.8) & (inst_freq < band_high * 1.2)
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient valid frequency samples")
    f_hz = float(np.median(inst_freq[valid_mask]))

    # Robust frequency dispersion: MAD → sigma
    freq_valid = inst_freq[valid_mask]
    mad_f = float(np.median(np.abs(freq_valid - np.median(freq_valid))))
    sigma_f_hz = mad_f * 1.4826  # MAD-to-sigma conversion factor

    peak_idx = int(np.argmax(envelope))
    snr_peak = float(envelope[peak_idx] / (np.std(envelope[:max(1, peak_idx // 2)]) + 1e-30))

    noise_floor = np.median(envelope) * 0.1 + 1e-30
    fit_mask = (np.arange(n) >= peak_idx) & (envelope > noise_floor)
    fit_indices = np.flatnonzero(fit_mask)
    if fit_indices.size < 5:
        raise ValueError(f"Insufficient samples for tau fit: {fit_indices.size}")

    t_fit = fit_indices.astype(float) / fs
    log_env = np.log(envelope[fit_indices])
    coeffs, cov = np.polyfit(t_fit - t_fit[0], log_env, 1, cov=True)
    gamma = -coeffs[0]
    var_gamma = float(cov[0, 0])

    if gamma <= 0:
        raise ValueError(f"Non-decaying signal: gamma={gamma:.4f}")

    tau_s = 1.0 / gamma
    Q = math.pi * f_hz * tau_s
    if Q <= 0 or not math.isfinite(Q):
        raise ValueError(f"Invalid Q={Q:.4f}")

    # --- Uncertainty propagation ---
    # tau = 1/gamma  =>  sigma_tau = sigma_gamma / gamma^2
    sigma_gamma = math.sqrt(var_gamma)
    sigma_tau_s = sigma_gamma / (gamma * gamma)

    # Q = pi * f * tau, assuming independence of f and tau:
    #   var_Q = (pi*tau)^2 * var_f + (pi*f)^2 * var_tau
    var_Q = (math.pi * tau_s) ** 2 * sigma_f_hz ** 2 \
          + (math.pi * f_hz) ** 2 * sigma_tau_s ** 2
    sigma_Q = math.sqrt(var_Q)

    # Log-space covariance for (ln f, ln Q):
    #   sigma_logf = sigma_f / f
    #   sigma_logQ = sigma_Q / Q
    #   cov(logf, logQ) ≈ 0  (independence assumption; valid as first gate)
    sigma_logf = sigma_f_hz / f_hz if f_hz > 0 else 0.0
    sigma_logQ = sigma_Q / Q if Q > 0 else 0.0
    cov_logf_logQ = 0.0  # independence assumption

    return {
        "f_hz": f_hz, "tau_s": tau_s, "Q": Q, "snr_peak": snr_peak,
        "sigma_f_hz": sigma_f_hz,
        "sigma_tau_s": sigma_tau_s,
        "sigma_Q": sigma_Q,
        "cov_logf_logQ": cov_logf_logQ,
    }


def bootstrap_ringdown_observables(
    strain: np.ndarray,
    fs: float,
    band_low: float,
    band_high: float,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Block bootstrap resampling to estimate uncertainties of (f, tau, Q).

    Method: block bootstrap with blocks of ~1 oscillation cycle.
    Resamples blocks of the strain with replacement, re-estimates observables.
    Reports median and std.

    Returns dict with:
        f_hz_median, f_hz_std, tau_s_median, tau_s_std, Q_median, Q_std,
        n_successful, n_failed, block_size,
        samples: {f_hz: [...], tau_s: [...], Q: [...]}
    """
    rng = np.random.default_rng(seed=seed)

    # (a) Initial estimate for block size calculation
    initial = estimate_ringdown_observables(strain, fs, band_low, band_high)
    f_estimate = initial["f_hz"]

    # (b) Block size ~ 1 oscillation cycle
    block_size = max(16, int(fs / f_estimate))
    n = len(strain)
    n_blocks = n // block_size

    _empty_result = {
        "f_hz_median": float("nan"), "f_hz_std": float("nan"),
        "tau_s_median": float("nan"), "tau_s_std": float("nan"),
        "Q_median": float("nan"), "Q_std": float("nan"),
        "n_successful": 0, "n_failed": 0, "block_size": block_size,
        "samples": {"f_hz": [], "tau_s": [], "Q": []},
    }
    if n_blocks == 0:
        return _empty_result

    # Number of blocks to draw so concatenated length >= n
    n_blocks_resample = -(-n // block_size)  # ceil(n / block_size)

    samples_f: list[float] = []
    samples_tau: list[float] = []
    samples_Q: list[float] = []
    n_failed = 0

    # (c) Bootstrap iterations
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, n_blocks, size=n_blocks_resample)
        resampled = np.concatenate(
            [strain[i * block_size:(i + 1) * block_size] for i in chosen]
        )[:n]

        try:
            est = estimate_ringdown_observables(resampled, fs, band_low, band_high)
            samples_f.append(est["f_hz"])
            samples_tau.append(est["tau_s"])
            samples_Q.append(est["Q"])
        except ValueError:
            n_failed += 1

    n_successful = len(samples_f)

    # (d) If too few successful, report NaN
    if n_successful < 10:
        _empty_result["n_successful"] = n_successful
        _empty_result["n_failed"] = n_failed
        return _empty_result

    # (e) Median and std of bootstrap distributions
    arr_f = np.array(samples_f)
    arr_tau = np.array(samples_tau)
    arr_Q = np.array(samples_Q)

    return {
        "f_hz_median": float(np.median(arr_f)),
        "f_hz_std": float(np.std(arr_f)),
        "tau_s_median": float(np.median(arr_tau)),
        "tau_s_std": float(np.std(arr_tau)),
        "Q_median": float(np.median(arr_Q)),
        "Q_std": float(np.std(arr_Q)),
        "n_successful": n_successful,
        "n_failed": n_failed,
        "block_size": block_size,
        "samples": {
            "f_hz": [float(x) for x in samples_f],
            "tau_s": [float(x) for x in samples_tau],
            "Q": [float(x) for x in samples_Q],
        },
    }


def estimate_ringdown_spectral(
    strain: np.ndarray,
    sample_rate: float,
    band_hz: tuple,
    nperseg: int | None = None,
) -> dict:
    """Spectral Lorentzian estimator for ringdown f, tau, Q.

    Computes PSD via Welch's method, fits a Lorentzian profile, and extracts
    the ringdown frequency (f_hz), decay time (tau_s), and quality factor (Q).

    Returns dict with keys:
        f_hz, tau_s, Q                  – point estimates
        sigma_f_hz, sigma_tau_s, sigma_Q – 1-sigma uncertainties from curve_fit pcov
        cov_logf_logQ                   – log-space covariance (independence approx)
        fit_success (bool)              – True if curve_fit converged with finite covariance
        fit_residual (float)            – normalized RMS residual of the Lorentzian fit

    On curve_fit failure (RuntimeError or infinite covariance), falls back to
    hilbert_envelope and sets fit_success=False.
    """
    import warnings
    from scipy.signal import butter, sosfilt, periodogram
    from scipy.optimize import curve_fit

    _nan_result = {
        "f_hz": float("nan"), "tau_s": float("nan"), "Q": float("nan"),
        "sigma_f_hz": float("nan"), "sigma_tau_s": float("nan"), "sigma_Q": float("nan"),
        "cov_logf_logQ": 0.0, "fit_success": False, "fit_residual": float("nan"),
    }

    n = len(strain)
    if n < 16:
        return dict(_nan_result)

    band_low, band_high = float(band_hz[0]), float(band_hz[1])
    nyquist = sample_rate / 2.0
    if band_high >= nyquist:
        band_high = nyquist * 0.95
    if band_low >= band_high:
        return dict(_nan_result)

    # Bandpass filter before computing PSD to focus on the ringdown band
    try:
        sos = butter(4, [band_low, band_high], btype="band", fs=sample_rate, output="sos")
        filtered = sosfilt(sos, strain)
    except Exception:
        filtered = strain

    # Zero-padded periodogram for sub-Hz frequency resolution.
    # Use rectangular (boxcar) window: the ringdown signal naturally decays
    # to near-zero, so the Hann window is NOT needed and would introduce
    # spectral broadening that biases tau.  Padding improves interpolation
    # without adding false averaging artefacts.
    # nperseg is accepted for API compat but controls only a floor on nfft.
    if nperseg is None:
        nperseg = min(n, int(sample_rate) // 4)
    nperseg = max(16, min(nperseg, n))

    # nfft: next power-of-2 above max(4*n, sample_rate*4) → ≤ 0.25 Hz/bin
    _nfft_min = max(n * 4, int(sample_rate) * 4)
    nfft = 1 << (_nfft_min - 1).bit_length()

    freqs_full, psd_full = periodogram(
        filtered, fs=sample_rate, window="boxcar", nfft=nfft, scaling="density",
    )

    # Restrict to the ringdown band
    band_mask = (freqs_full >= band_low) & (freqs_full <= band_high)
    if np.sum(band_mask) < 5:
        return dict(_nan_result)

    f_band = freqs_full[band_mask]
    psd_band = psd_full[band_mask]

    # Lorentzian model: L(f) = A / ((f - f0)^2 + gamma^2)
    # where gamma = 1 / (2 * pi * tau)  (half-width at half-maximum in Hz)
    # This is the one-sided PSD Lorentzian for a damped sinusoid with
    # amplitude decay time tau and Q = pi * f0 * tau.
    def lorentzian(f, A, f0, tau):
        gamma = 1.0 / (2.0 * math.pi * tau)
        return A / ((f - f0) ** 2 + gamma ** 2)

    # Seed: use the peak bin as f0 initial guess (better than mid-band)
    peak_idx_band = int(np.argmax(psd_band))
    f0_init = float(f_band[peak_idx_band])
    tau_init = 0.05  # 50 ms, typical BBH ringdown
    gamma_init = 1.0 / (2.0 * math.pi * tau_init)  # ≈ 3.18 Hz

    # Focus the fit on ±10 × HWHM_init around the peak to avoid noise-floor
    # domination (the vast off-peak noise degrades OLS if left unconstrained).
    fit_half_width = max(gamma_init * 10.0, 30.0)  # at least ±30 Hz
    peak_region = np.abs(f_band - f0_init) <= fit_half_width
    if np.sum(peak_region) < 5:
        peak_region = np.ones(len(f_band), dtype=bool)

    f_fit = f_band[peak_region]
    psd_fit = psd_band[peak_region]

    A_init = float(psd_fit[int(np.argmax(psd_fit))]) * gamma_init ** 2

    # chi-squared sigma weighting: var(periodogram_bin) ≈ S_true(f)^2
    # → sigma_i = PSD_i makes the fit minimise relative (not absolute) errors
    sigma_fit = np.maximum(psd_fit, np.max(psd_fit) * 1e-6)

    p0 = [A_init, f0_init, tau_init]
    bounds_low  = [0.0,    band_low,  1e-4]
    bounds_high = [np.inf, band_high, 1.0]

    fit_success = False
    f0_fit = float("nan")
    tau_fit = float("nan")
    sigma_f = float("nan")
    sigma_tau = float("nan")
    fit_residual = float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                lorentzian, f_fit, psd_fit,
                p0=p0,
                bounds=(bounds_low, bounds_high),
                sigma=sigma_fit,
                absolute_sigma=True,
                maxfev=10000,
            )

        A_fit, f0_fit, tau_fit = popt

        if np.all(np.isfinite(pcov)):
            sigma_f   = float(np.sqrt(pcov[1, 1]))
            sigma_tau = float(np.sqrt(pcov[2, 2]))
            fit_success = True

            psd_model = lorentzian(f_fit, *popt)
            denom = float(np.mean(psd_fit) ** 2) + 1e-100
            fit_residual = float(np.mean((psd_fit - psd_model) ** 2) / denom)
    except RuntimeError:
        pass

    if not fit_success:
        # Fallback: use hilbert_envelope and flag the failure
        try:
            result = estimate_ringdown_observables(
                strain, sample_rate, band_hz[0], band_hz[1]
            )
            result["fit_success"] = False
            result["fit_residual"] = float("nan")
            return result
        except Exception:
            return dict(_nan_result)

    Q_fit = math.pi * f0_fit * tau_fit

    # Uncertainty propagation for Q = pi * f0 * tau (assuming independence)
    sigma_Q = math.sqrt(
        (math.pi * tau_fit) ** 2 * sigma_f ** 2
        + (math.pi * f0_fit) ** 2 * sigma_tau ** 2
    )

    return {
        "f_hz":         float(f0_fit),
        "tau_s":        float(tau_fit),
        "Q":            float(Q_fit),
        "sigma_f_hz":   sigma_f,
        "sigma_tau_s":  sigma_tau,
        "sigma_Q":      sigma_Q,
        "cov_logf_logQ": 0.0,
        "fit_success":  True,
        "fit_residual": fit_residual,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: estimate f, tau, Q")
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--band-low", type=float, default=150.0)
    ap.add_argument("--band-high", type=float, default=400.0)
    ap.add_argument("--method", default="spectral_lorentzian",
                    choices=["hilbert_envelope", "spectral_lorentzian"],
                    help="Estimator to use: 'spectral_lorentzian' (default, unbiased) or "
                         "'hilbert_envelope' (legacy, ~13%% f bias, ~39%% Q bias)")
    ap.add_argument("--n-bootstrap", type=int, default=200,
                    help="Number of bootstrap resamples for uncertainty estimation (0=skip, "
                         "ignored when --method=spectral_lorentzian)")
    args = ap.parse_args()

    run_id = args.run_id or args.run
    if not run_id:
        ap.error("one of --run or --run-id is required")
    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())

    ctx = init_stage(run_id, STAGE, params={
        "band_low_hz": args.band_low, "band_high_hz": args.band_high,
        "method": args.method, "n_bootstrap": args.n_bootstrap,
    })

    # Discover detector files
    upstream_dir = ctx.run_dir / "s2_ringdown_window" / "outputs"
    det_files: dict[str, Path] = {}
    for cand in ("H1_rd.npz", "L1_rd.npz", "V1_rd.npz"):
        p = upstream_dir / cand
        if p.exists():
            det_files[cand.split("_")[0]] = p
    if not det_files:
        abort(ctx, f"No detector files in {upstream_dir}")

    check_inputs(ctx, det_files)

    # Load window metadata
    wm_path = upstream_dir / "window_meta.json"
    window_meta: dict[str, Any] = {}
    if wm_path.exists():
        with open(wm_path, "r", encoding="utf-8") as f:
            window_meta = json.load(f)

    try:
        per_detector: dict[str, Any] = {}
        valid: list[dict[str, float]] = []
        spectral_fallbacks: list[str] = []  # detectors that fell back to hilbert

        for det, path in det_files.items():
            data = np.load(path)
            strain = np.asarray(data["strain"], dtype=np.float64)
            fs = float(np.asarray(data["sample_rate_hz"]).flat[0])
            if strain.ndim != 1 or not np.all(np.isfinite(strain)):
                abort(ctx, f"{det}: invalid strain")
            try:
                if args.method == "spectral_lorentzian":
                    est = estimate_ringdown_spectral(
                        strain, fs, (args.band_low, args.band_high)
                    )
                    if not est.get("fit_success", True):
                        spectral_fallbacks.append(det)
                    # Validate that we got usable estimates
                    if not (math.isfinite(est.get("f_hz", float("nan")))
                            and math.isfinite(est.get("tau_s", float("nan")))):
                        raise ValueError(
                            f"spectral_lorentzian returned non-finite estimates: "
                            f"f={est.get('f_hz')}, tau={est.get('tau_s')}"
                        )
                else:  # hilbert_envelope
                    est = estimate_ringdown_observables(strain, fs, args.band_low, args.band_high)

                per_detector[det] = est

                # Bootstrap uncertainty estimation (hilbert_envelope only)
                if args.method == "hilbert_envelope" and args.n_bootstrap > 0:
                    boot = bootstrap_ringdown_observables(
                        strain, fs, args.band_low, args.band_high,
                        n_bootstrap=args.n_bootstrap,
                    )
                    _bstd_f = boot["f_hz_std"]
                    _bstd_tau = boot["tau_s_std"]
                    _bstd_Q = boot["Q_std"]
                    per_detector[det]["uncertainty"] = {
                        "method": "block_bootstrap",
                        "n_bootstrap": args.n_bootstrap,
                        "n_successful": boot["n_successful"],
                        "n_failed": boot["n_failed"],
                        "block_size": boot["block_size"],
                        "f_hz_std": _bstd_f if math.isfinite(_bstd_f) else None,
                        "tau_s_std": _bstd_tau if math.isfinite(_bstd_tau) else None,
                        "Q_std": _bstd_Q if math.isfinite(_bstd_Q) else None,
                    }
                    per_detector[det]["bootstrap_samples"] = boot["samples"]

                valid.append(per_detector[det])
            except ValueError as exc:
                per_detector[det] = {"error": str(exc)}

        if not valid:
            abort(ctx, "No detector produced a valid estimate")

        weights = np.array([e.get("snr_peak", 1.0) for e in valid])
        weights = weights / weights.sum()
        combined_f = float(sum(w * e["f_hz"] for w, e in zip(weights, valid)))
        combined_tau = float(sum(w * e["tau_s"] for w, e in zip(weights, valid)))
        combined_Q = math.pi * combined_f * combined_tau

        # Combined uncertainties (analytic): propagate per-detector variances
        # through the SNR-weighted average (var_comb = sum(w_i^2 * var_i))
        var_f_comb = float(sum(
            w ** 2 * e.get("sigma_f_hz", 0.0) ** 2
            for w, e in zip(weights, valid)
        ))
        var_tau_comb = float(sum(
            w ** 2 * e.get("sigma_tau_s", 0.0) ** 2
            for w, e in zip(weights, valid)
        ))
        sigma_f_comb = math.sqrt(var_f_comb)
        sigma_tau_comb = math.sqrt(var_tau_comb)

        # Q = pi * f * tau  =>  var_Q = (pi*tau)^2 var_f + (pi*f)^2 var_tau
        var_Q_comb = (math.pi * combined_tau) ** 2 * var_f_comb \
                   + (math.pi * combined_f) ** 2 * var_tau_comb
        sigma_Q_comb = math.sqrt(var_Q_comb)

        # Log-space covariance matrix for downstream (s4 Paso 2)
        sigma_logf = sigma_f_comb / combined_f if combined_f > 0 else 0.0
        sigma_logQ = sigma_Q_comb / combined_Q if combined_Q > 0 else 0.0
        cov_logf_logQ = 0.0  # independence assumption

        # Canonical (modern) aliases expected by downstream contract-first audits.
        sigma_lnf = sigma_logf
        sigma_lnQ = sigma_logQ
        if sigma_logf > 0 and sigma_logQ > 0:
            r = cov_logf_logQ / (sigma_logf * sigma_logQ)
            r = float(max(min(r, 1.0 - 1e-12), -1.0 + 1e-12))
        else:
            r = 0.0

        # Build combined dict with uncertainties
        combined_dict: dict[str, Any] = {
            "f_hz": combined_f, "tau_s": combined_tau, "Q": combined_Q,
        }
        if args.method == "hilbert_envelope" and args.n_bootstrap > 0:
            # Propagate per-detector bootstrap stds via SNR weights
            boot_var_f = 0.0
            boot_var_tau = 0.0
            boot_var_Q = 0.0
            any_valid_boot = False
            for w, e in zip(weights, valid):
                unc = e.get("uncertainty", {})
                std_f = unc.get("f_hz_std")
                if std_f is not None and math.isfinite(std_f):
                    boot_var_f += w ** 2 * std_f ** 2
                    boot_var_tau += w ** 2 * unc["tau_s_std"] ** 2
                    boot_var_Q += w ** 2 * unc["Q_std"] ** 2
                    any_valid_boot = True

            if any_valid_boot:
                combined_sigma_f = math.sqrt(boot_var_f)
                combined_sigma_tau = math.sqrt(boot_var_tau)
                combined_sigma_Q = math.sqrt(boot_var_Q)
                combined_dict["sigma_f_hz"] = combined_sigma_f
                combined_dict["sigma_tau_s"] = combined_sigma_tau
                combined_dict["sigma_Q"] = combined_sigma_Q
                combined_dict["sigma_log_f"] = (
                    combined_sigma_f / combined_f if combined_f > 0 else None
                )
                combined_dict["sigma_log_Q"] = (
                    combined_sigma_Q / combined_Q if combined_Q > 0 else None
                )
            else:
                combined_dict["sigma_f_hz"] = None
                combined_dict["sigma_tau_s"] = None
                combined_dict["sigma_Q"] = None
                combined_dict["sigma_log_f"] = None
                combined_dict["sigma_log_Q"] = None
        elif args.method == "spectral_lorentzian":
            # Use analytic uncertainties from the combined propagation
            combined_dict["sigma_f_hz"] = sigma_f_comb if math.isfinite(sigma_f_comb) else None
            combined_dict["sigma_tau_s"] = sigma_tau_comb if math.isfinite(sigma_tau_comb) else None
            combined_dict["sigma_Q"] = sigma_Q_comb if math.isfinite(sigma_Q_comb) else None
            combined_dict["sigma_log_f"] = (
                sigma_logf if (combined_f > 0 and math.isfinite(sigma_logf)) else None
            )
            combined_dict["sigma_log_Q"] = (
                sigma_logQ if (combined_Q > 0 and math.isfinite(sigma_logQ)) else None
            )

        estimates: dict[str, Any] = {
            "schema_version": "mvp_estimates_v2",
            "event_id": window_meta.get("event_id", "unknown"),
            "method": args.method,
            "band_hz": [args.band_low, args.band_high],
            "combined": combined_dict,
            "combined_uncertainty": {
                "sigma_f_hz": sigma_f_comb,
                "sigma_tau_s": sigma_tau_comb,
                "sigma_Q": sigma_Q_comb,

                # Legacy keys (kept for backward compatibility)
                "cov_logf_logQ": cov_logf_logQ,
                "sigma_logf": sigma_logf,
                "sigma_logQ": sigma_logQ,

                # Modern canonical keys
                "sigma_lnf": sigma_lnf,
                "sigma_lnQ": sigma_lnQ,
                "r": float(r),
            },
            "per_detector": per_detector,
            "n_detectors_valid": len(valid),
        }
        if args.method == "hilbert_envelope" and args.n_bootstrap > 0:
            estimates["bootstrap"] = {
                "n_requested": args.n_bootstrap,
                "method": "block_bootstrap",
            }
        est_path = ctx.outputs_dir / "estimates.json"
        write_json_atomic(est_path, estimates)

        extra_summary: dict[str, Any] = {}
        if spectral_fallbacks:
            extra_summary["warnings"] = [
                f"spectral_lorentzian curve_fit failed for {det}; "
                f"hilbert_envelope fallback used"
                for det in spectral_fallbacks
            ]

        finalize(ctx, artifacts={"estimates": est_path},
                 results=estimates["combined"],
                 extra_summary=extra_summary if extra_summary else None)
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
