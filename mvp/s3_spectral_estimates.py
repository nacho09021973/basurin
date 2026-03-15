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



def _load_measured_psd(psd_path: Path) -> dict[str, Any]:
    payload = json.loads(psd_path.read_text(encoding="utf-8"))
    freqs = np.asarray(payload.get("frequencies_hz", []), dtype=np.float64)
    if freqs.ndim != 1 or freqs.size < 2 or not np.all(np.isfinite(freqs)):
        raise ValueError(f"Invalid frequencies_hz in measured PSD: {psd_path}")
    if not np.all(np.diff(freqs) > 0):
        raise ValueError(f"frequencies_hz must be strictly increasing: {psd_path}")
    return payload


def _whiten_with_measured_psd(
    strain: np.ndarray,
    fs: float,
    *,
    detector: str,
    psd_payload: dict[str, Any],
) -> np.ndarray:
    det_key = f"psd_{detector.upper()}"
    raw_psd = psd_payload.get(det_key)
    if not isinstance(raw_psd, list):
        raise KeyError(det_key)

    psd_freqs = np.asarray(psd_payload.get("frequencies_hz", []), dtype=np.float64)
    psd_vals = np.asarray(raw_psd, dtype=np.float64)
    if psd_vals.shape != psd_freqs.shape:
        raise ValueError(f"{det_key} length mismatch with frequencies_hz")
    if np.any(~np.isfinite(psd_vals)):
        raise ValueError(f"{det_key} contains non-finite PSD values")
    psd_vals = np.maximum(psd_vals, 1e-30)

    n = strain.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    interp = np.interp(freqs, psd_freqs, psd_vals, left=psd_vals[0], right=psd_vals[-1])
    interp = np.maximum(interp, 1e-30)

    spec = np.fft.rfft(strain)
    white_spec = spec / np.sqrt(interp)
    return np.fft.irfft(white_spec, n=n)


def _lorentzian(nu: np.ndarray, A: float, f0: float, gamma: float, C: float) -> np.ndarray:
    """Single-sided Lorentzian model (peak only, for frequency range > 0).

    L(ν) = A / [(ν - f0)² + (γ/2)²] + C
    """
    return A / ((nu - f0) ** 2 + (gamma / 2.0) ** 2) + C


def _estimate_peak_snr(
    strain: np.ndarray,
    fs: float,
    *,
    band_low: float,
    band_high: float,
) -> float:
    from scipy.signal import periodogram

    n = strain.size
    if n < 16:
        return 0.0

    nyquist = fs / 2.0
    high = min(float(band_high), nyquist * 0.95)
    low = float(band_low)
    if low >= high:
        return 0.0

    nfft_floor = max(n * 4, int(fs) * 4)
    nfft = 1 << (nfft_floor - 1).bit_length()
    freqs, psd = periodogram(
        strain,
        fs=fs,
        window="boxcar",
        nfft=nfft,
        scaling="density",
    )
    mask = (freqs >= low) & (freqs <= high)
    if int(np.count_nonzero(mask)) < 5:
        return 0.0

    psd_band = np.asarray(psd[mask], dtype=np.float64)
    noise_floor = float(np.median(psd_band)) + 1e-30
    return float(np.max(psd_band) / noise_floor)


def estimate_spectral_observables(
    strain: np.ndarray,
    fs: float,
    band_low: float = 150.0,
    band_high: float = 400.0,
) -> dict[str, Any]:
    """Estimate f, tau, Q via the robust shared spectral Lorentzian fitter.

    This stage historically carried a simplified fitter that diverged from the
    canonical implementation in ``s3_ringdown_estimates`` and produced biased
    outputs on short windows.  Keep the stage-local schema, but delegate the
    actual fit to the shared implementation so both spectral branches stay
    numerically aligned.
    """
    from mvp.s3_ringdown_estimates import estimate_ringdown_spectral

    result = estimate_ringdown_spectral(
        strain,
        fs,
        (band_low, band_high),
        kerr_centered_band=False,
    )

    fit_converged = bool(result.get("fit_success", False))
    f_hz = float(result.get("f_hz", float("nan")))
    spectral_band = result.get("spectral_band")
    df_floor_hz = result.get("df_floor_hz")
    edge_warning = False
    if (
        fit_converged
        and isinstance(spectral_band, dict)
        and isinstance(df_floor_hz, (int, float))
        and math.isfinite(float(df_floor_hz))
        and math.isfinite(f_hz)
    ):
        band_low_eff = spectral_band.get("f_low_hz")
        band_high_eff = spectral_band.get("f_high_hz")
        if isinstance(band_low_eff, (int, float)) and isinstance(band_high_eff, (int, float)):
            edge_warning = (
                (f_hz - float(band_low_eff)) <= float(df_floor_hz)
                or (float(band_high_eff) - f_hz) <= float(df_floor_hz)
            )

    mapped: dict[str, Any] = {
        "f_hz": f_hz,
        "tau_s": float(result.get("tau_s", float("nan"))),
        "Q": float(result.get("Q", float("nan"))),
        "snr_peak": _estimate_peak_snr(
            strain,
            fs,
            band_low=band_low,
            band_high=band_high,
        ) if fit_converged else 0.0,
        "sigma_f_hz": float(result.get("sigma_f_hz", float("nan"))),
        "sigma_tau_s": float(result.get("sigma_tau_s", float("nan"))),
        "sigma_Q": float(result.get("sigma_Q", float("nan"))),
        "cov_logf_logQ": float(result.get("cov_logf_logQ", 0.0)),
        "fit_converged": fit_converged,
        "edge_warning": edge_warning,
        "fit": result.get("fit"),
        "fit_residual": result.get("fit_residual"),
        "fit_failure_reason": result.get("fit_failure_reason"),
        "fit_degenerate": result.get("fit_degenerate"),
        "spectral_band": spectral_band,
        "sigma_f_hz_raw": result.get("sigma_f_hz_raw"),
        "df_floor_hz": result.get("df_floor_hz"),
        "sigma_floor_applied": result.get("sigma_floor_applied"),
    }
    return mapped


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
    ap.add_argument("--psd-path", default=None, help="Path to measured_psd.json from extract_psd.py")
    args = ap.parse_args()

    run_id = args.run_id or args.run
    if not run_id:
        ap.error("one of --run or --run-id is required")
    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())

    ctx = init_stage(run_id, STAGE, params={
        "band_low_hz": args.band_low, "band_high_hz": args.band_high,
        "method": "spectral_lorentzian", "n_bootstrap": args.n_bootstrap,
        "psd_path": args.psd_path,
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

    psd_payload: dict[str, Any] | None = None
    if args.psd_path:
        psd_file = Path(args.psd_path)
        if psd_file.exists():
            psd_payload = _load_measured_psd(psd_file)
        else:
            print(
                f"[{STAGE}] WARNING: --psd-path {args.psd_path!r} not found; "
                "falling back to internal Welch PSD",
                flush=True,
            )

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
                detector_psd_source = "internal_welch"
                if psd_payload is not None:
                    try:
                        strain = _whiten_with_measured_psd(
                            strain,
                            fs,
                            detector=det,
                            psd_payload=psd_payload,
                        )
                        detector_psd_source = "external_measured_psd"
                    except (KeyError, ValueError) as exc:
                        print(
                            f"[{STAGE}] WARNING: measured PSD unusable for detector={det} ({exc}); "
                            "falling back to internal Welch",
                            flush=True,
                        )

                est = estimate_spectral_observables(
                    strain, fs, args.band_low, args.band_high
                )
                est["psd_source"] = detector_psd_source
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
                per_detector[det] = {"error": str(exc), "psd_source": detector_psd_source}

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

        combined_dict: dict[str, Any] = {
            "f_hz": combined_f, "tau_s": combined_tau, "Q": combined_Q,
        }

        sigma_f_out = sigma_f_comb
        sigma_tau_out = sigma_tau_comb
        sigma_Q_out = sigma_Q_comb

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
                sigma_f_out = combined_sigma_f
                sigma_tau_out = combined_sigma_tau
                sigma_Q_out = combined_sigma_Q

        sigma_logf = sigma_f_out / combined_f if combined_f > 0 else 0.0
        sigma_logQ = sigma_Q_out / combined_Q if combined_Q > 0 else 0.0
        cov_logf_logQ = 0.0

        sigma_lnf = sigma_logf
        sigma_lnQ = sigma_logQ
        if sigma_logf > 0 and sigma_logQ > 0:
            r = cov_logf_logQ / (sigma_logf * sigma_logQ)
            r = float(max(min(r, 1.0 - 1e-12), -1.0 + 1e-12))
        else:
            r = 0.0

        combined_dict["sigma_f_hz"] = sigma_f_out if math.isfinite(sigma_f_out) else None
        combined_dict["sigma_tau_s"] = sigma_tau_out if math.isfinite(sigma_tau_out) else None
        combined_dict["sigma_Q"] = sigma_Q_out if math.isfinite(sigma_Q_out) else None
        combined_dict["sigma_log_f"] = (
            sigma_logf if (combined_f > 0 and math.isfinite(sigma_logf)) else None
        )
        combined_dict["sigma_log_Q"] = (
            sigma_logQ if (combined_Q > 0 and math.isfinite(sigma_logQ)) else None
        )

        # Build output (schema compatible with estimates.json for s4 consumption)
        psd_source = "internal_welch"
        if valid and all(v.get("psd_source") == "external_measured_psd" for v in valid):
            psd_source = "external_measured_psd"

        estimates: dict[str, Any] = {
            "schema_version": "mvp_spectral_estimates_v1",
            "event_id": window_meta.get("event_id", "unknown"),
            "method": "spectral_lorentzian",
            "band_hz": [args.band_low, args.band_high],
            "combined": combined_dict,
            "combined_uncertainty": {
                "sigma_f_hz": sigma_f_out,
                "sigma_tau_s": sigma_tau_out,
                "sigma_Q": sigma_Q_out,
                "cov_logf_logQ": cov_logf_logQ,
                "sigma_logf": sigma_logf,
                "sigma_logQ": sigma_logQ,
                "sigma_lnf": sigma_lnf,
                "sigma_lnQ": sigma_lnQ,
                "r": float(r),
            },
            "per_detector": per_detector,
            "n_detectors_valid": len(valid),
            "psd_source": psd_source,
        }
        if args.n_bootstrap > 0:
            estimates["bootstrap"] = {
                "n_requested": args.n_bootstrap,
                "method": "block_bootstrap",
            }

        est_path = ctx.outputs_dir / "spectral_estimates.json"
        write_json_atomic(est_path, estimates)

        finalize(
            ctx,
            artifacts={"spectral_estimates": est_path},
            results=estimates["combined"],
            extra_summary={"psd_source": psd_source},
        )
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
