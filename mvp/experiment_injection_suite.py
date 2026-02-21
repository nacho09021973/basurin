#!/usr/bin/env python3
"""Injection suite: validate estimators via synthetic ringdown recovery.

CLI:
    python mvp/experiment_injection_suite.py --run <run_id> \
        [--n-f 10] [--n-Q 8] [--snr-values 5,8,10,15,20,30] \
        [--seed 42] [--band-low 150] [--band-high 400]

Outputs:
    runs/<run>/experiment/INJECTION_SUITE_V1/injection_results.json

Method:
    480 injections (10 × 8 × 6) with known (f_true, Q_true, SNR):
    - Synthetic ringdown: h(t) = A·exp(-πft/Q)·cos(2πft)
    - Gaussian colored noise with simplified aLIGO PSD
    - Recover with both Hilbert and spectral estimators
    - Compute bias, coverage, and recovery rate statistics
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import resolve_out_root, sha256_file, utc_now_iso, write_json_atomic
from mvp.s3_ringdown_estimates import estimate_ringdown_observables
from mvp.s3_spectral_estimates import estimate_spectral_observables

FS = 4096.0       # Sample rate for synthetic injections (Hz)
DURATION = 0.5    # Seconds of signal (ringdown + context)
F_REF = 200.0     # PSD reference frequency


def _psd_aligo_shape(f: np.ndarray) -> np.ndarray:
    """Simplified aLIGO PSD shape (same as s6): (f0/f)^4 + 2 + 2*(f/f0)^2."""
    x = f / F_REF
    x = np.where(x <= 0, 1e-10, x)
    return x ** (-4) + 2.0 + 2.0 * x ** 2


def _generate_colored_noise(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Gaussian noise colored by simplified aLIGO PSD."""
    white = rng.standard_normal(n)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    psd_shape = _psd_aligo_shape(np.where(freqs <= 0, freqs[1] if len(freqs) > 1 else 1.0, freqs))
    # Color by sqrt(PSD) in amplitude; zero out DC to avoid astronomical spike
    fft_colored = fft_white * np.sqrt(psd_shape)
    fft_colored[0] = 0.0  # zero DC: prevents (200/1e-6)^4 ≈ 1.6e33 spike
    colored = np.fft.irfft(fft_colored, n=n)
    return colored


def _make_injection(
    f_true: float,
    Q_true: float,
    snr_true: float,
    fs: float,
    duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic ringdown + colored noise at given SNR.

    h(t) = A·exp(-πf/Q·t)·cos(2πft)
    SNR = A / sigma_noise  (rough optimal SNR)
    """
    n = int(duration * fs)
    t = np.arange(n) / fs

    tau = Q_true / (math.pi * f_true)
    signal = np.exp(-t / tau) * np.cos(2.0 * math.pi * f_true * t)

    noise = _generate_colored_noise(n, fs, rng)
    # Normalize noise to unit std before scaling
    noise_std = float(np.std(noise)) + 1e-30
    noise = noise / noise_std

    # Scale signal so that signal RMS / noise RMS = snr_true
    signal_rms = float(np.sqrt(np.mean(signal ** 2))) + 1e-30
    A = snr_true * (1.0 / signal_rms)

    return A * signal + noise


def _run_hilbert(
    strain: np.ndarray, fs: float, band_low: float, band_high: float,
) -> dict[str, Any]:
    """Run Hilbert estimator on injection, return result dict."""
    try:
        est = estimate_ringdown_observables(strain, fs, band_low, band_high)
        return {
            "f_est": est["f_hz"],
            "Q_est": est["Q"],
            "tau_est": est["tau_s"],
            "sigma_f": est["sigma_f_hz"],
            "sigma_Q": est["sigma_Q"],
            "converged": True,
        }
    except Exception as exc:
        return {
            "f_est": float("nan"), "Q_est": float("nan"), "tau_est": float("nan"),
            "sigma_f": float("nan"), "sigma_Q": float("nan"),
            "converged": False, "error": str(exc),
        }


def _run_spectral(
    strain: np.ndarray, fs: float, band_low: float, band_high: float,
) -> dict[str, Any]:
    """Run spectral (Lorentzian) estimator on injection, return result dict."""
    try:
        est = estimate_spectral_observables(strain, fs, band_low, band_high)
        if not est.get("fit_converged", True):
            return {
                "f_est": float("nan"), "Q_est": float("nan"), "tau_est": float("nan"),
                "sigma_f": float("nan"), "sigma_Q": float("nan"),
                "converged": False, "error": "Lorentzian fit did not converge",
            }
        return {
            "f_est": est["f_hz"],
            "Q_est": est["Q"],
            "tau_est": est["tau_s"],
            "sigma_f": est["sigma_f_hz"],
            "sigma_Q": est["sigma_Q"],
            "converged": True,
        }
    except Exception as exc:
        return {
            "f_est": float("nan"), "Q_est": float("nan"), "tau_est": float("nan"),
            "sigma_f": float("nan"), "sigma_Q": float("nan"),
            "converged": False, "error": str(exc),
        }


def _bias_dict(est: dict[str, Any], f_true: float, Q_true: float) -> dict[str, float]:
    """Compute bias fields."""
    f_est = est["f_est"]
    Q_est = est["Q_est"]
    if math.isfinite(f_est) and math.isfinite(Q_est):
        delta_f = f_est - f_true
        delta_Q = Q_est - Q_true
        delta_f_rel = delta_f / f_true
        delta_Q_rel = delta_Q / Q_true
    else:
        delta_f = float("nan")
        delta_Q = float("nan")
        delta_f_rel = float("nan")
        delta_Q_rel = float("nan")
    return {
        "delta_f": delta_f, "delta_Q": delta_Q,
        "delta_f_rel": delta_f_rel, "delta_Q_rel": delta_Q_rel,
    }


def _compute_summary(results: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """Compute aggregate statistics for a method (hilbert or spectral)."""
    key = method  # "hilbert" or "spectral"

    delta_f_rels = []
    delta_Q_rels = []
    coverage_f_68 = []
    coverage_f_95 = []
    coverage_Q_68 = []
    coverage_Q_95 = []
    n_recovered = 0
    n_total = 0

    for r in results:
        n_total += 1
        est = r[key]
        bias = r[f"bias_{key}"]

        if not est["converged"]:
            continue
        if not math.isfinite(est["f_est"]) or not math.isfinite(est["Q_est"]):
            continue

        f_est = est["f_est"]
        Q_est = est["Q_est"]
        f_true = r["f_true"]
        Q_true = r["Q_true"]
        delta_f_rel = bias["delta_f_rel"]
        delta_Q_rel = bias["delta_Q_rel"]

        # Recovery criterion: |Δf/f| < 10% AND |ΔQ/Q| < 30%
        if abs(delta_f_rel) < 0.10 and abs(delta_Q_rel) < 0.30:
            n_recovered += 1

        delta_f_rels.append(delta_f_rel)
        delta_Q_rels.append(delta_Q_rel)

        # Coverage: is |Δf| within 1σ / 2σ?
        sigma_f = est["sigma_f"]
        sigma_Q = est["sigma_Q"]
        if math.isfinite(sigma_f) and sigma_f > 0:
            abs_delta_f = abs(f_est - f_true)
            coverage_f_68.append(1.0 if abs_delta_f < sigma_f else 0.0)
            coverage_f_95.append(1.0 if abs_delta_f < 2.0 * sigma_f else 0.0)
        if math.isfinite(sigma_Q) and sigma_Q > 0:
            abs_delta_Q = abs(Q_est - Q_true)
            coverage_Q_68.append(1.0 if abs_delta_Q < sigma_Q else 0.0)
            coverage_Q_95.append(1.0 if abs_delta_Q < 2.0 * sigma_Q else 0.0)

    recovery_rate = n_recovered / n_total if n_total > 0 else 0.0
    median_bias_f = float(np.median(delta_f_rels)) if delta_f_rels else float("nan")
    median_bias_Q = float(np.median(delta_Q_rels)) if delta_Q_rels else float("nan")
    cov_68_f = float(np.mean(coverage_f_68)) if coverage_f_68 else float("nan")
    cov_95_f = float(np.mean(coverage_f_95)) if coverage_f_95 else float("nan")
    cov_68_Q = float(np.mean(coverage_Q_68)) if coverage_Q_68 else float("nan")
    cov_95_Q = float(np.mean(coverage_Q_95)) if coverage_Q_95 else float("nan")

    return {
        "recovery_rate": recovery_rate,
        "n_recovered": n_recovered,
        "n_total": n_total,
        "median_bias_f_rel": median_bias_f,
        "median_bias_Q_rel": median_bias_Q,
        "coverage_68_f": cov_68_f,
        "coverage_95_f": cov_95_f,
        "coverage_68_Q": cov_68_Q,
        "coverage_95_Q": cov_95_Q,
    }


def _quality_gate(summary: dict[str, Any], method: str) -> dict[str, Any]:
    """Check if estimator passes quality gate."""
    rr = summary["recovery_rate"]
    mbf = abs(summary["median_bias_f_rel"])
    cov68 = summary["coverage_68_f"]

    gate_pass = (
        rr > 0.80 and
        mbf < 0.05 and
        math.isfinite(cov68) and 0.55 <= cov68 <= 0.80
    )

    failures = []
    if rr <= 0.80:
        failures.append(f"recovery_rate={rr:.3f} ≤ 0.80")
    if mbf >= 0.05:
        failures.append(f"|median_bias_f_rel|={mbf:.3f} ≥ 0.05")
    if not (math.isfinite(cov68) and 0.55 <= cov68 <= 0.80):
        failures.append(f"coverage_68_f={cov68:.3f} not in [0.55, 0.80]")

    return {
        "method": method,
        "gate": "PASS" if gate_pass else "FAIL",
        "failures": failures,
    }


def run_injection_suite(
    run_id: str,
    n_f: int = 10,
    n_Q: int = 8,
    snr_values: list[float] | None = None,
    seed: int = 42,
    band_low: float = 150.0,
    band_high: float = 400.0,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Run the injection suite and return the results dict."""
    if out_root is None:
        out_root = resolve_out_root("runs")
    if snr_values is None:
        snr_values = [5.0, 8.0, 10.0, 15.0, 20.0, 30.0]

    # Grid
    f_values = list(np.logspace(math.log10(150.0), math.log10(350.0), n_f))
    Q_values = list(np.logspace(math.log10(2.0), math.log10(15.0), n_Q))

    n_injections = n_f * n_Q * len(snr_values)
    print(
        f"[injection_suite] Running {n_injections} injections "
        f"({n_f}×{n_Q}×{len(snr_values)})",
        flush=True,
    )

    rng = np.random.default_rng(seed=seed)
    results: list[dict[str, Any]] = []

    for f_true in f_values:
        for Q_true in Q_values:
            for snr_true in snr_values:
                strain = _make_injection(
                    f_true=f_true, Q_true=Q_true, snr_true=snr_true,
                    fs=FS, duration=DURATION, rng=rng,
                )

                hilbert_est = _run_hilbert(strain, FS, band_low, band_high)
                spectral_est = _run_spectral(strain, FS, band_low, band_high)

                bias_h = _bias_dict(hilbert_est, f_true, Q_true)
                bias_s = _bias_dict(spectral_est, f_true, Q_true)

                results.append({
                    "f_true": float(f_true),
                    "Q_true": float(Q_true),
                    "tau_true": float(Q_true / (math.pi * f_true)),
                    "snr_true": float(snr_true),
                    "hilbert": hilbert_est,
                    "spectral": spectral_est,
                    "bias_hilbert": bias_h,
                    "bias_spectral": bias_s,
                })

    # Compute summaries
    summary_hilbert = _compute_summary(results, "hilbert")
    summary_spectral = _compute_summary(results, "spectral")

    gate_hilbert = _quality_gate(summary_hilbert, "hilbert")
    gate_spectral = _quality_gate(summary_spectral, "spectral")

    output: dict[str, Any] = {
        "schema_version": "mvp_injection_suite_v1",
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "seed": seed,
        "fs_hz": FS,
        "duration_s": DURATION,
        "band_hz": [band_low, band_high],
        "n_injections": n_injections,
        "grid": {
            "f_values_hz": [float(x) for x in f_values],
            "Q_values": [float(x) for x in Q_values],
            "snr_values": [float(x) for x in snr_values],
            "n_f": n_f, "n_Q": n_Q, "n_snr": len(snr_values),
        },
        "results": results,
        "summary": {
            "hilbert": summary_hilbert,
            "spectral": summary_spectral,
        },
        "quality_gates": {
            "hilbert": gate_hilbert,
            "spectral": gate_spectral,
        },
    }

    # Write output
    run_dir = out_root / run_id
    exp_dir = run_dir / "experiment" / "INJECTION_SUITE_V1"
    exp_dir.mkdir(parents=True, exist_ok=True)

    out_path = exp_dir / "injection_results.json"
    write_json_atomic(out_path, output)

    # Manifest with SHA256
    manifest = {
        "schema_version": "mvp_manifest_v1",
        "stage": "experiment_injection_suite",
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "artifacts": [
            {
                "path": str(out_path.relative_to(run_dir)),
                "sha256": sha256_file(out_path),
            }
        ],
    }
    write_json_atomic(exp_dir / "manifest.json", manifest)

    print(f"[injection_suite] Hilbert gate: {gate_hilbert['gate']}", flush=True)
    print(f"[injection_suite]   recovery={summary_hilbert['recovery_rate']:.3f}, "
          f"bias_f={summary_hilbert['median_bias_f_rel']:.3f}, "
          f"cov68={summary_hilbert['coverage_68_f']:.3f}", flush=True)
    print(f"[injection_suite] Spectral gate: {gate_spectral['gate']}", flush=True)
    print(f"[injection_suite]   recovery={summary_spectral['recovery_rate']:.3f}, "
          f"bias_f={summary_spectral['median_bias_f_rel']:.3f}, "
          f"cov68={summary_spectral['coverage_68_f']:.3f}", flush=True)
    print(f"[injection_suite] Output: {out_path}", flush=True)

    return output


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Injection suite: validate estimators via synthetic ringdown recovery"
    )
    ap.add_argument("--run", required=True, help="Run ID")
    ap.add_argument("--n-f", type=int, default=10, help="Number of frequency grid points")
    ap.add_argument("--n-Q", type=int, default=8, help="Number of Q grid points")
    ap.add_argument("--snr-values", type=str, default="5,8,10,15,20,30",
                    help="Comma-separated SNR values")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band-low", type=float, default=150.0)
    ap.add_argument("--band-high", type=float, default=400.0)
    args = ap.parse_args()

    snr_values = [float(x) for x in args.snr_values.split(",") if x.strip()]

    try:
        run_injection_suite(
            run_id=args.run,
            n_f=args.n_f,
            n_Q=args.n_Q,
            snr_values=snr_values,
            seed=args.seed,
            band_low=args.band_low,
            band_high=args.band_high,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
