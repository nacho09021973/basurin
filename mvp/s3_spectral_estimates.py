#!/usr/bin/env python3
"""MVP Stage 3 (variant): Estimate ringdown observables via Lorentzian spectral fit.

CLI:
    python mvp/s3_spectral_estimates.py --run <run_id> \
        [--band-low 150] [--band-high 400] [--n-bootstrap 200]

Inputs:  runs/<run>/s2_ringdown_window/outputs/{H1,L1,V1}_rd.npz
Outputs: runs/<run>/s3_spectral_estimates/outputs/spectral_estimates.json

Method:
    A damped sinusoid h(t) = A·exp(-t/τ)·cos(2πft + φ) has a power spectrum
    that is a Lorentzian:

        PSD(ν) ∝ 1/[(ν-f)² + (1/(4πτ))²] + 1/[(ν+f)² + (1/(4πτ))²]

    The peak frequency gives f, and the FWHM of the peak gives τ via:
        FWHM = 1/(2πτ)  =>  τ = 1/(π·FWHM)
        Q = π·f·τ = f/FWHM

    Advantages over Hilbert:
    - Less sensitive to phase noise contamination
    - FWHM-based τ is more robust than envelope slope fit
    - Uncertainties come directly from curve_fit covariance (calibrated)
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

STAGE = "s3_spectral_estimates"


def _lorentzian(nu: np.ndarray, A: float, f0: float, gamma: float, C: float) -> np.ndarray:
    """Single-sided Lorentzian model (peak only, for frequency range > 0).

    L(ν) = A / [(ν - f0)² + (γ/2)²] + C
    """
    return A / ((nu - f0) ** 2 + (gamma / 2.0) ** 2) + C


def estimate_spectral_observables(
    strain: np.ndarray,
    fs: float,
    band_low: float = 150.0,
    band_high: float = 400.0,
) -> dict[str, Any]:
    """Estimate f, tau, Q via Lorentzian fit to the power spectral density.

    Returns point estimates and uncertainty fields compatible with s3 schema:
      f_hz, tau_s, Q, snr_peak       – point estimates
      sigma_f_hz, sigma_tau_s, sigma_Q – 1-sigma uncertainties from fit
      cov_logf_logQ                   – covariance in (ln f, ln Q) space
      fit_converged                   – True if curve_fit succeeded
    """
    from scipy.signal import welch
    from scipy.optimize import curve_fit

    n = strain.size
    if n < 32:
        raise ValueError(f"Strain too short: {n} samples")

    nyquist = fs / 2.0
    if band_high >= nyquist:
        band_high = nyquist * 0.95
    if band_low >= band_high:
        raise ValueError(f"Invalid band: [{band_low}, {band_high}] Hz")

    # Welch PSD: nperseg ~ 50ms segments
    nperseg = min(n, int(0.05 * fs))
    nperseg = max(nperseg, 16)

    freqs, psd = welch(strain, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                       window="hann", scaling="density")

    # Restrict to band
    mask = (freqs >= band_low) & (freqs <= band_high)
    if np.sum(mask) < 5:
        raise ValueError(f"Too few frequency bins in band [{band_low}, {band_high}] Hz")

    freqs_band = freqs[mask]
    psd_band = psd[mask]

    # Find peak
    peak_idx = int(np.argmax(psd_band))
    f_peak = float(freqs_band[peak_idx])
    psd_peak = float(psd_band[peak_idx])
    psd_median = float(np.median(psd_band))

    # Check peak not at edge (warn via flag, not abort)
    edge_warning = (peak_idx < 2) or (peak_idx > len(freqs_band) - 3)

    # Initial parameters for Lorentzian fit
    # γ_init: rough FWHM estimate from half-max crossing
    half_max = (psd_peak + psd_median) / 2.0
    above_half = freqs_band[psd_band > half_max]
    if len(above_half) >= 2:
        gamma_init = float(above_half[-1] - above_half[0])
        gamma_init = max(gamma_init, 1.0)
    else:
        gamma_init = 10.0

    A_init = psd_peak * (gamma_init / 2.0) ** 2  # A = PSD_peak * (γ/2)²
    p0 = [A_init, f_peak, gamma_init, psd_median * 0.1]

    # Bounds: keep f0 in band, γ > 0, A > 0, C >= 0
    bounds_low = [0.0, band_low, 0.1, 0.0]
    bounds_high = [np.inf, band_high, (band_high - band_low), np.inf]

    fit_converged = True
    try:
        popt, pcov = curve_fit(
            _lorentzian, freqs_band, psd_band,
            p0=p0, bounds=(bounds_low, bounds_high),
            maxfev=10000,
        )
        A_fit, f0_fit, gamma_fit, C_fit = popt

        # Check covariance is valid
        if not np.all(np.isfinite(pcov)):
            fit_converged = False
    except (RuntimeError, ValueError):
        fit_converged = False

    if not fit_converged:
        # Return NaN estimates with flag
        return {
            "f_hz": float("nan"), "tau_s": float("nan"), "Q": float("nan"),
            "snr_peak": 0.0,
            "sigma_f_hz": float("nan"), "sigma_tau_s": float("nan"),
            "sigma_Q": float("nan"),
            "cov_logf_logQ": 0.0,
            "fit_converged": False,
            "edge_warning": edge_warning,
        }

    # Extract physical observables
    f_hz = float(f0_fit)
    gamma = float(gamma_fit)  # FWHM
    tau_s = 1.0 / (math.pi * gamma)
    Q = f_hz / gamma  # = π·f·τ

    if Q <= 0 or not math.isfinite(Q):
        raise ValueError(f"Invalid Q={Q:.4f} from fit")

    # Uncertainties from covariance matrix
    # popt = [A, f0, gamma, C]  =>  indices: f0=1, gamma=2
    sigma_f_hz = float(math.sqrt(abs(pcov[1, 1])))
    sigma_gamma = float(math.sqrt(abs(pcov[2, 2])))

    # Propagation: tau = 1/(π·γ)  =>  sigma_tau = sigma_gamma / (π·γ²)
    sigma_tau_s = sigma_gamma / (math.pi * gamma ** 2)

    # Q = f/γ  =>  var_Q = Q²·[(σ_f/f)² + (σ_γ/γ)²]
    sigma_Q = Q * math.sqrt((sigma_f_hz / f_hz) ** 2 + (sigma_gamma / gamma) ** 2)

    # Log-space covariance: sigma_logf, sigma_logQ
    sigma_logf = sigma_f_hz / f_hz if f_hz > 0 else 0.0
    sigma_logQ = sigma_Q / Q if Q > 0 else 0.0

    # Covariance in log-space from fit covariance (non-zero correlation)
    # cov(ln f, ln Q) = cov(ln f, ln f - ln γ)
    #   = var(ln f) - cov(ln f, ln γ)
    # In linearized: cov(ln f, ln γ) ≈ cov(f, γ) / (f * γ)
    cov_f_gamma = float(pcov[1, 2])
    cov_logf_logQ = (sigma_logf ** 2 - cov_f_gamma / (f_hz * gamma)
                     if (f_hz > 0 and gamma > 0) else 0.0)

    # SNR estimate: peak height relative to noise floor
    noise_floor = float(np.median(psd_band)) + 1e-30
    snr_peak = float(psd_peak / noise_floor)

    return {
        "f_hz": f_hz, "tau_s": tau_s, "Q": Q,
        "snr_peak": snr_peak,
        "sigma_f_hz": sigma_f_hz,
        "sigma_tau_s": sigma_tau_s,
        "sigma_Q": sigma_Q,
        "cov_logf_logQ": cov_logf_logQ,
        "fit_converged": True,
        "edge_warning": edge_warning,
        "_fit_params": {
            "A": float(A_fit), "f0": float(f0_fit),
            "gamma_fwhm": float(gamma_fit), "C": float(C_fit),
        },
    }


def bootstrap_spectral_observables(
    strain: np.ndarray,
    fs: float,
    band_low: float,
    band_high: float,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Block bootstrap for spectral estimator.

    Resamples blocks of the strain, re-estimates PSD and fits Lorentzian.
    Reports median and std.

    Returns dict with:
        f_hz_median, f_hz_std, tau_s_median, tau_s_std, Q_median, Q_std,
        n_successful, n_failed, block_size,
        samples: {f_hz: [...], tau_s: [...], Q: [...]}
    """
    rng = np.random.default_rng(seed=seed)

    # Initial estimate for block size
    try:
        initial = estimate_spectral_observables(strain, fs, band_low, band_high)
        if not initial["fit_converged"]:
            raise ValueError("Initial spectral fit did not converge")
        f_estimate = initial["f_hz"]
    except ValueError:
        # Fallback block size
        f_estimate = (band_low + band_high) / 2.0

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

    n_blocks_resample = -(-n // block_size)  # ceil

    samples_f: list[float] = []
    samples_tau: list[float] = []
    samples_Q: list[float] = []
    n_failed = 0

    for _ in range(n_bootstrap):
        chosen = rng.integers(0, n_blocks, size=n_blocks_resample)
        resampled = np.concatenate(
            [strain[i * block_size:(i + 1) * block_size] for i in chosen]
        )[:n]

        try:
            est = estimate_spectral_observables(resampled, fs, band_low, band_high)
            if not est["fit_converged"] or not math.isfinite(est["f_hz"]):
                n_failed += 1
                continue
            samples_f.append(est["f_hz"])
            samples_tau.append(est["tau_s"])
            samples_Q.append(est["Q"])
        except ValueError:
            n_failed += 1

    n_successful = len(samples_f)

    if n_successful < 10:
        _empty_result["n_successful"] = n_successful
        _empty_result["n_failed"] = n_failed
        return _empty_result

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


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: Lorentzian spectral estimator")
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default=None,
                    help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--band-low", type=float, default=150.0)
    ap.add_argument("--band-high", type=float, default=400.0)
    ap.add_argument("--n-bootstrap", type=int, default=200,
                    help="Number of bootstrap resamples (0=skip)")
    args = ap.parse_args()

    run_id = args.run_id or args.run
    if not run_id:
        ap.error("one of --run or --run-id is required")
    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())

    ctx = init_stage(run_id, STAGE, params={
        "band_low_hz": args.band_low, "band_high_hz": args.band_high,
        "method": "spectral_lorentzian", "n_bootstrap": args.n_bootstrap,
    })

    # Discover detector files (same pattern as s3)
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

        for det, path in det_files.items():
            data = np.load(path)
            strain = np.asarray(data["strain"], dtype=np.float64)
            fs = float(np.asarray(data["sample_rate_hz"]).flat[0])
            if strain.ndim != 1 or not np.all(np.isfinite(strain)):
                abort(ctx, f"{det}: invalid strain")
            try:
                est = estimate_spectral_observables(
                    strain, fs, args.band_low, args.band_high
                )
                per_detector[det] = est

                if not est["fit_converged"]:
                    per_detector[det]["error"] = "Lorentzian fit did not converge"
                    continue

                # Bootstrap uncertainty estimation
                if args.n_bootstrap > 0:
                    boot = bootstrap_spectral_observables(
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
            abort(ctx, "No detector produced a valid spectral estimate")

        # SNR-weighted combination
        weights = np.array([max(e.get("snr_peak", 1.0), 1e-6) for e in valid])
        weights = weights / weights.sum()
        combined_f = float(sum(w * e["f_hz"] for w, e in zip(weights, valid)))
        combined_tau = float(sum(w * e["tau_s"] for w, e in zip(weights, valid)))
        combined_Q = math.pi * combined_f * combined_tau

        # Combined uncertainties via SNR-weighted variance propagation
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

        var_Q_comb = (math.pi * combined_tau) ** 2 * var_f_comb \
                   + (math.pi * combined_f) ** 2 * var_tau_comb
        sigma_Q_comb = math.sqrt(var_Q_comb)

        # Log-space covariance
        sigma_logf = sigma_f_comb / combined_f if combined_f > 0 else 0.0
        sigma_logQ = sigma_Q_comb / combined_Q if combined_Q > 0 else 0.0
        cov_logf_logQ = 0.0

        sigma_lnf = sigma_logf
        sigma_lnQ = sigma_logQ
        if sigma_logf > 0 and sigma_logQ > 0:
            r = cov_logf_logQ / (sigma_logf * sigma_logQ)
            r = float(max(min(r, 1.0 - 1e-12), -1.0 + 1e-12))
        else:
            r = 0.0

        combined_dict: dict[str, Any] = {
            "f_hz": combined_f, "tau_s": combined_tau, "Q": combined_Q,
        }

        if args.n_bootstrap > 0:
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

        # Build output (schema compatible with estimates.json for s4 consumption)
        estimates: dict[str, Any] = {
            "schema_version": "mvp_spectral_estimates_v1",
            "event_id": window_meta.get("event_id", "unknown"),
            "method": "spectral_lorentzian",
            "band_hz": [args.band_low, args.band_high],
            "combined": combined_dict,
            "combined_uncertainty": {
                "sigma_f_hz": sigma_f_comb,
                "sigma_tau_s": sigma_tau_comb,
                "sigma_Q": sigma_Q_comb,
                "cov_logf_logQ": cov_logf_logQ,
                "sigma_logf": sigma_logf,
                "sigma_logQ": sigma_logQ,
                "sigma_lnf": sigma_lnf,
                "sigma_lnQ": sigma_lnQ,
                "r": float(r),
            },
            "per_detector": per_detector,
            "n_detectors_valid": len(valid),
        }
        if args.n_bootstrap > 0:
            estimates["bootstrap"] = {
                "n_requested": args.n_bootstrap,
                "method": "block_bootstrap",
            }

        est_path = ctx.outputs_dir / "spectral_estimates.json"
        write_json_atomic(est_path, estimates)

        finalize(ctx, artifacts={"spectral_estimates": est_path},
                 results=estimates["combined"])
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
