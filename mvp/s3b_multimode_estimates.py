#!/usr/bin/env python3
"""Stage s3b_multimode_estimates: canonical multi-mode estimates (220,221)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage
from mvp.s3_ringdown_estimates import estimate_ringdown_observables, estimate_ringdown_spectral

STAGE = "s3b_multimode_estimates"
TARGET_MODES = [
    {"mode": [2, 2, 0], "label": "220"},
    {"mode": [2, 2, 1], "label": "221"},
]
MIN_BOOTSTRAP_SAMPLES = 128
SIGMA_DET_EPS = 1e-12
SIGMA_COND_MAX = 1e12

# --- model_comparison numerical guards (documented in output conventions block) ---
_RSS_FLOOR = 1e-300         # floor applied to rss/n inside log; prevents -inf on perfect fits
_N_MIN_BIC_MARGIN = 2       # n must exceed k + this margin for BIC to be interpretable
_DELTA_BIC_TIE_EPS = 1e-10  # |delta_bic| < eps → prefer simpler 1-mode (deterministic tie-break)


def compute_covariance(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must be shape (n,2)")
    if samples.shape[0] < 2:
        raise ValueError("need >=2 samples for covariance")
    return np.cov(samples, rowvar=False, ddof=1)


def is_valid_sigma_matrix(sigma: np.ndarray) -> bool:
    if sigma.shape != (2, 2) or not np.all(np.isfinite(sigma)):
        return False

    det = float(np.linalg.det(sigma))
    if not math.isfinite(det) or abs(det) <= SIGMA_DET_EPS:
        return False

    try:
        cond = float(np.linalg.cond(sigma))
    except np.linalg.LinAlgError:
        return False
    if not math.isfinite(cond) or cond > SIGMA_COND_MAX:
        return False

    try:
        inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(np.isfinite(inv)))


def covariance_gate(
    sigma: np.ndarray,
    *,
    sigma_floor: float = 1e-4,
    sigma_ceiling: float = 2.0,
    corr_limit: float = 0.999,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if sigma.shape != (2, 2) or not np.all(np.isfinite(sigma)):
        return False, ["Sigma_non_finite"]
    if sigma[0, 0] <= 0 or sigma[1, 1] <= 0:
        reasons.append("Sigma_non_positive_diag")

    s0 = float(math.sqrt(max(sigma[0, 0], 0.0)))
    s1 = float(math.sqrt(max(sigma[1, 1], 0.0)))
    if not (sigma_floor <= s0 <= sigma_ceiling):
        reasons.append("sigma_lnf_out_of_bounds")
    if not (sigma_floor <= s1 <= sigma_ceiling):
        reasons.append("sigma_lnQ_out_of_bounds")

    det = float(np.linalg.det(sigma))
    if det <= 0:
        reasons.append("Sigma_not_invertible")

    if s0 > 0 and s1 > 0:
        r = float(sigma[0, 1] / (s0 * s1))
        if abs(r) >= corr_limit:
            reasons.append("corr_too_high")
    else:
        reasons.append("sigma_zero")

    return len(reasons) == 0, reasons


def _resolve_input_path(run_dir: Path, stage_dir: Path, raw_path: str) -> Path:
    cand = Path(raw_path)
    if cand.is_absolute() and cand.exists():
        return cand
    rel_to_run = run_dir / cand
    if rel_to_run.exists():
        return rel_to_run
    rel_to_stage = stage_dir / cand
    if rel_to_stage.exists():
        return rel_to_stage
    return rel_to_run


def _discover_s2_npz(run_dir: Path) -> Path:
    stage_dir = run_dir / "s2_ringdown_window"
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing s2 manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("corrupt s2 manifest: missing artifacts map")

    candidates: list[tuple[str, Path]] = []
    for key, raw in artifacts.items():
        if not isinstance(key, str) or not isinstance(raw, str):
            continue
        if key in {"H1_rd", "L1_rd"} or raw.endswith("_rd.npz"):
            candidates.append((key, _resolve_input_path(run_dir, stage_dir, raw)))

    if not candidates:
        raise RuntimeError("missing canonical NPZ rd output from s2")

    by_key = {k: p for k, p in candidates}
    if "H1_rd" in by_key:
        return by_key["H1_rd"]
    if "L1_rd" in by_key:
        return by_key["L1_rd"]
    return sorted(p for _, p in candidates)[0]


def _discover_s2_window_meta(run_dir: Path) -> Path | None:
    stage_dir = run_dir / "s2_ringdown_window"
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    raw = artifacts.get("window_meta")
    if not isinstance(raw, str):
        return None
    path = _resolve_input_path(run_dir, stage_dir, raw)
    if not path.exists():
        return None
    return path


def _resolve_s3_estimates(run_dir: Path, raw_path: str | None) -> Path:
    if raw_path:
        path = _resolve_input_path(run_dir, run_dir / STAGE, raw_path)
        if path.exists():
            return path
    return run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"


def _coerce_band(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    low = float(raw[0])
    high = float(raw[1])
    if not (math.isfinite(low) and math.isfinite(high) and low > 0 and high > low):
        raise RuntimeError("invalid s3 estimates: invalid band_hz values")
    return low, high


def _load_s3_band(path: Path, *, window_meta: dict[str, Any] | None) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cand in (payload.get("band_hz"), payload.get("results", {}).get("bandpass_hz")):
        band = _coerce_band(cand)
        if band is not None:
            return band

    meta_band = _coerce_band(window_meta.get("band_hz") if isinstance(window_meta, dict) else None)
    if meta_band is not None:
        return meta_band

    raise RuntimeError(
        "invalid s3 estimates: missing band_hz (and no band_hz in s2 window_meta fallback)"
    )


def compute_robust_stability(samples: list[tuple[float, float]]) -> dict[str, float | None]:
    if not samples:
        return {
            "lnf_p10": None,
            "lnf_p50": None,
            "lnf_p90": None,
            "lnQ_p10": None,
            "lnQ_p50": None,
            "lnQ_p90": None,
            "lnf_span": None,
            "lnQ_span": None,
        }

    arr = np.asarray(samples, dtype=float)
    lnf = arr[:, 0]
    lnq = arr[:, 1]
    lnf_p10, lnf_p50, lnf_p90 = np.percentile(lnf, [10, 50, 90])
    lnq_p10, lnq_p50, lnq_p90 = np.percentile(lnq, [10, 50, 90])

    return {
        "lnf_p10": float(lnf_p10),
        "lnf_p50": float(lnf_p50),
        "lnf_p90": float(lnf_p90),
        "lnQ_p10": float(lnq_p10),
        "lnQ_p50": float(lnq_p50),
        "lnQ_p90": float(lnq_p90),
        "lnf_span": float(lnf_p90 - lnf_p10),
        "lnQ_span": float(lnq_p90 - lnq_p10),
    }


def _load_signal_from_npz(path: Path, *, window_meta: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    try:
        data = np.load(path)
    except Exception as exc:
        raise RuntimeError(f"corrupt s2 NPZ: cannot read {path}: {exc}") from exc

    if "strain" not in data:
        raise RuntimeError("corrupt s2 NPZ: missing strain")
    signal = np.asarray(data["strain"], dtype=float)

    fs: float | None = None
    for key in ("sample_rate_hz", "fs"):
        if key in data:
            fs = float(np.asarray(data[key]).flat[0])
            break
    if fs is None and window_meta:
        for key in ("sample_rate_hz", "fs"):
            if key in window_meta:
                fs = float(window_meta[key])
                break

    if signal.ndim != 1 or signal.size < 16 or not np.all(np.isfinite(signal)):
        raise RuntimeError("corrupt s2 NPZ: invalid strain")
    if fs is None or not math.isfinite(fs) or fs <= 0:
        raise RuntimeError("corrupt s2 NPZ: missing/invalid sample rate")
    return signal, fs


def _bootstrap_mode_log_samples(
    signal: np.ndarray,
    fs: float,
    estimator: Callable[[np.ndarray, float], dict[str, float]],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    n = signal.size
    block_len = max(64, n // 20)
    n_blocks = int(math.ceil(n / block_len))

    samples: list[list[float]] = []
    failed = 0
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        resampled = np.concatenate([signal[s:s + block_len] for s in starts])[:n]
        try:
            est = estimator(resampled, fs)
            f_hz = float(est["f_hz"])
            q = float(est["Q"])
            if f_hz > 0 and q > 0 and math.isfinite(f_hz) and math.isfinite(q):
                samples.append([math.log(f_hz), math.log(q)])
            else:
                failed += 1
        except Exception:
            failed += 1

    if not samples:
        return np.empty((0, 2), dtype=float), failed
    return np.asarray(samples, dtype=float), failed


def _bootstrap_block_size(n_samples: int) -> int:
    return max(64, n_samples // 20)


def _estimate_220(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    return estimate_ringdown_observables(signal, fs, band_low, band_high)


def _estimate_spectral(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    est = estimate_ringdown_spectral(signal, fs, (band_low, band_high))
    return {
        "f_hz": float(est["f_hz"]),
        "Q": float(est["Q"]),
        "tau_s": float(est["tau_s"]),
    }


def _split_mode_bands(*, band_low: float, band_high: float) -> tuple[tuple[float, float], tuple[float, float]]:
    mid = 0.5 * (band_low + band_high)
    eps = 1e-6
    band_220 = (band_low, max(mid - eps, band_low + eps))
    band_221 = (min(mid + eps, band_high - eps), band_high)
    return band_220, band_221


def _estimate_220_spectral(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    band_220, _ = _split_mode_bands(band_low=band_low, band_high=band_high)
    return _estimate_spectral(signal, fs, band_low=band_220[0], band_high=band_220[1])


def _template_220(signal: np.ndarray, fs: float, est220: dict[str, float]) -> np.ndarray:
    from numpy.linalg import lstsq

    t = np.arange(signal.size, dtype=float) / fs
    f = float(est220["f_hz"])
    tau = float(est220["tau_s"])
    env = np.exp(-t / tau)
    w = 2.0 * math.pi * f
    b1 = env * np.cos(w * t)
    b2 = env * np.sin(w * t)
    X = np.vstack([b1, b2]).T
    coeffs, *_ = lstsq(X, signal, rcond=None)
    return (X @ coeffs).astype(float)


def _estimate_221_from_signal(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    est220 = _estimate_220(signal, fs, band_low=band_low, band_high=band_high)
    residual = signal - _template_220(signal, fs, est220)
    return estimate_ringdown_observables(residual, fs, band_low, band_high)


def _estimate_221_spectral_two_pass(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    est220 = _estimate_220_spectral(signal, fs, band_low=band_low, band_high=band_high)
    residual = signal - _template_220(signal, fs, est220)
    _, band_221 = _split_mode_bands(band_low=band_low, band_high=band_high)
    return _estimate_spectral(residual, fs, band_low=band_221[0], band_high=band_221[1])


def _mode_null(
    label: str,
    mode: list[int],
    n_bootstrap: int,
    seed: int,
    stability: dict[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "label": label,
        "ln_f": None,
        "ln_Q": None,
        "Sigma": None,
        "fit": {
            "method": method,
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "stability": stability,
        },
    }


def evaluate_mode(
    signal: np.ndarray,
    fs: float,
    *,
    label: str,
    mode: list[int],
    estimator: Callable[[np.ndarray, float], dict[str, float]],
    n_bootstrap: int,
    seed: int,
    min_valid_fraction: float = 0.0,
    cv_threshold: float | None = None,
    max_lnf_span: float = 1.0,
    max_lnq_span: float = 1.0,
    min_point_samples: int = 2,
    min_point_valid_fraction: float = 0.0,
    method: str = "hilbert_peakband",
) -> tuple[dict[str, Any], list[str], bool]:
    flags: list[str] = []
    stability = {
        "valid_fraction": 0.0,
        "n_successful": 0,
        "n_failed": int(n_bootstrap),
        "cv_f": None,
        "cv_Q": None,
        **compute_robust_stability([]),
    }
    n_samples = int(signal.size)
    if n_samples <= 0 or n_samples < MIN_BOOTSTRAP_SAMPLES:
        flags.append("bootstrap_high_nonpositive")
        stability["message"] = "window too short after offset"
        return _mode_null(label, mode, n_bootstrap, seed, stability, method=method), sorted(flags), False

    block_size = _bootstrap_block_size(n_samples)
    if block_size <= 0 or block_size >= n_samples:
        flags.append("bootstrap_block_invalid")
        stability["message"] = "window too short after offset"
        return _mode_null(label, mode, n_bootstrap, seed, stability, method=method), sorted(flags), False

    rng = np.random.default_rng(int(seed))

    try:
        samples, n_failed = _bootstrap_mode_log_samples(signal, fs, estimator, n_bootstrap=n_bootstrap, rng=rng)
    except ValueError as exc:
        msg = str(exc)
        if "high <= 0" in msg or "low >= high" in msg:
            flags.append("bootstrap_high_nonpositive")
            stability["message"] = "window too short after offset"
            return _mode_null(label, mode, n_bootstrap, seed, stability, method=method), sorted(flags), False
        raise
    sigma_invalid_flag = f"{label}_Sigma_invalid"
    if samples.ndim != 2 or samples.shape[1] != 2:
        flags.append(sigma_invalid_flag)
        return _mode_null(label, mode, n_bootstrap, seed, stability, method=method), sorted(set(flags)), False

    valid_mask = np.all(np.isfinite(samples), axis=1)
    dropped_invalid = int(samples.shape[0] - int(np.count_nonzero(valid_mask)))
    if dropped_invalid > 0:
        flags.append(f"{label}_invalid_bootstrap_samples")
    samples = samples[valid_mask]
    n_failed += dropped_invalid

    if method == "spectral_two_pass" and samples.size > 0:
        # spectral_two_pass second coordinate is ln(tau_s); convert to ln(Q)
        # using Q = pi * f * tau_s to keep downstream stability/Sigma semantics.
        samples = samples.copy()
        samples[:, 1] = math.log(math.pi) + samples[:, 0] + samples[:, 1]

    valid_fraction = float(samples.shape[0] / n_bootstrap) if n_bootstrap > 0 else 0.0
    stability["valid_fraction"] = valid_fraction
    stability["n_successful"] = int(samples.shape[0])
    stability["n_failed"] = int(n_failed)
    stability.update(compute_robust_stability([tuple(s) for s in samples.tolist()]))

    can_materialize_point = (
        samples.shape[0] >= int(min_point_samples)
        and valid_fraction >= float(min_point_valid_fraction)
        and stability.get("lnf_p50") is not None
        and stability.get("lnQ_p50") is not None
    )
    if can_materialize_point:
        ln_f = float(stability["lnf_p50"])
        ln_q = float(stability["lnQ_p50"])
    else:
        ln_f = None
        ln_q = None
        flags.append(f"{label}_no_point_estimate")

    if samples.shape[0] < 2:
        flags.append(f"{label}_bootstrap_insufficient")
        flags.append(sigma_invalid_flag)
        return _mode_null(label, mode, n_bootstrap, seed, stability, method=method), sorted(flags), False

    sigma: np.ndarray | None = None
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] < int(min_point_samples):
        flags.append(sigma_invalid_flag)
    else:
        sigma_candidate = compute_covariance(samples)
        if is_valid_sigma_matrix(sigma_candidate):
            sigma = sigma_candidate
        else:
            flags.append(sigma_invalid_flag)

    sigma_invertible = sigma is not None

    ok = True
    if not sigma_invertible:
        ok = False

    if valid_fraction < min_valid_fraction:
        flags.append(f"{label}_valid_fraction_low")
        ok = False

    lnf_span = stability.get("lnf_span")
    lnq_span = stability.get("lnQ_span")
    if lnf_span is None or not math.isfinite(float(lnf_span)):
        flags.append(f"{label}_lnf_span_invalid")
        ok = False
    elif float(lnf_span) > max_lnf_span:
        flags.append(f"{label}_lnf_span_explosive")
        ok = False

    if lnq_span is None or not math.isfinite(float(lnq_span)):
        flags.append(f"{label}_lnQ_span_invalid")
        ok = False
    elif float(lnq_span) > max_lnq_span:
        flags.append(f"{label}_lnQ_span_explosive")
        ok = False

    cv_f = None
    cv_q = None
    if cv_threshold is not None:
        f_vals = np.exp(samples[:, 0])
        q_vals = np.exp(samples[:, 1])
        cv_f = float(np.std(f_vals, ddof=1) / (np.mean(f_vals) + 1e-30))
        cv_q = float(np.std(q_vals, ddof=1) / (np.mean(q_vals) + 1e-30))
        stability["cv_f"] = cv_f
        stability["cv_Q"] = cv_q
        if cv_f > cv_threshold:
            flags.append(f"{label}_cv_f_explosive")
        if cv_q > cv_threshold:
            flags.append(f"{label}_cv_Q_explosive")

    if ln_f is None or ln_q is None:
        ok = False
    elif not (math.isfinite(ln_f) and math.isfinite(ln_q)):
        flags.append(f"{label}_non_finite_log")
        ok = False

    if sigma is None:
        mode_payload = _mode_null(label, mode, n_bootstrap, seed, stability, method=method)
        mode_payload["ln_f"] = ln_f
        mode_payload["ln_Q"] = ln_q
    else:
        mode_payload = {
            "mode": mode,
            "label": label,
            "ln_f": ln_f,
            "ln_Q": ln_q,
            "Sigma": [[float(sigma[0, 0]), float(sigma[0, 1])], [float(sigma[1, 0]), float(sigma[1, 1])]],
            "fit": {
                "method": method,
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_seed": int(seed),
                "stability": stability,
            },
        }

    return mode_payload, sorted(set(flags)), ok


def compute_model_comparison(
    signal: np.ndarray,
    fs: float,
    mode_220: dict[str, Any],
    mode_221: dict[str, Any],
    ok_220: bool,
    ok_221: bool,
    *,
    delta_bic_threshold: float = -10.0,
) -> dict[str, Any]:
    """BIC-based model comparison: 1-mode (220) vs 2-mode (220+221).

    BIC formula (Gaussian i.i.d. noise, profiled variance):
        BIC = k * ln(n) + n * ln(rss / n)

    k convention: 4 per mode (Acos, Asin, f, tau). Documented in output.
    delta_bic = bic_2mode - bic_1mode; two_mode_preferred iff delta_bic < threshold.

    Numerical guards (all documented in returned conventions block):
    - rss/n is clamped to _RSS_FLOOR before log to prevent -inf on perfect fits.
    - BIC is only computed (valid_bic_*mode=True) when n > k + _N_MIN_BIC_MARGIN.
    - Tie-breaking: |delta_bic| < _DELTA_BIC_TIE_EPS => prefer 1-mode deterministically.
    """
    from numpy.linalg import lstsq

    n = int(signal.size)
    k_1mode = 4
    k_2mode = 8

    # Conventions block: all semantic choices documented for audit / reproducibility.
    conventions: dict[str, Any] = {
        "delta_bic_definition": "bic_2mode - bic_1mode",
        "two_mode_threshold": float(delta_bic_threshold),
        "design_matrix_columns": ["220_cos", "220_sin", "221_cos", "221_sin"],
        "k_definition": {
            "k_1mode": k_1mode,
            "k_2mode": k_2mode,
            "counts": {
                "per_mode": ["Acos", "Asin", "f", "tau"],
                "one_mode": ["Acos_220", "Asin_220", "f_220", "tau_220"],
                "two_mode": [
                    "Acos_220", "Asin_220", "f_220", "tau_220",
                    "Acos_221", "Asin_221", "f_221", "tau_221",
                ],
            },
        },
        "bic_formula": "k*ln(n) + n*ln(rss/n)",
        "rss_floor": _RSS_FLOOR,
        "n_min_bic_margin": _N_MIN_BIC_MARGIN,
        "delta_bic_tie_eps": _DELTA_BIC_TIE_EPS,
        "tie_break_rule": "|ΔBIC|<eps => ΔBIC=0.0 and prefer 1-mode",
    }

    trace_base: dict[str, Any] = {
        "fit_method": "lstsq_amplitudes",
        "bic_formula": "k*ln(n) + n*ln(rss/n)",
        "k_convention": "k=4 per mode (Acos, Asin, f, tau)",
        "mode_220_label": mode_220.get("label"),
        "mode_221_label": mode_221.get("label"),
        "rss_floored_1mode": False,
        "rss_floored_2mode": False,
    }

    def _null_result(*, valid_1mode: bool, valid_2mode: bool, trace_extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": "model_comparison_v1",
            "n_samples": n,
            "k_1mode": k_1mode,
            "k_2mode": k_2mode,
            "rss_1mode": None,
            "rss_2mode": None,
            "bic_1mode": None,
            "bic_2mode": None,
            "delta_bic": None,
            "thresholds": {"two_mode_preferred_delta_bic": delta_bic_threshold},
            "conventions": conventions,
            "decision": {"two_mode_preferred": None},
            "valid_1mode": valid_1mode,
            "valid_2mode": valid_2mode,
            "valid_bic_1mode": False,
            "valid_bic_2mode": False,
            "trace": {**trace_base, **(trace_extra or {})},
        }

    if n <= 0 or not math.isfinite(float(fs)) or float(fs) <= 0.0:
        return _null_result(
            valid_1mode=False,
            valid_2mode=False,
            trace_extra={"invalid_signal_or_fs": True},
        )

    t = np.arange(n, dtype=float) / fs

    # --- Validate 220 point estimates ---
    valid_1mode = (
        ok_220
        and mode_220.get("ln_f") is not None
        and mode_220.get("ln_Q") is not None
        and math.isfinite(float(mode_220["ln_f"]))
        and math.isfinite(float(mode_220["ln_Q"]))
    )
    if not valid_1mode:
        return _null_result(valid_1mode=False, valid_2mode=False)

    ln_f220 = float(mode_220["ln_f"])
    ln_q220 = float(mode_220["ln_Q"])
    f220 = math.exp(ln_f220)
    q220 = math.exp(ln_q220)
    tau220 = q220 / (math.pi * f220)
    w220 = 2.0 * math.pi * f220
    env220 = np.exp(-t / tau220)
    col_220c = env220 * np.cos(w220 * t)
    col_220s = env220 * np.sin(w220 * t)
    X1 = np.vstack([col_220c, col_220s]).T

    coeffs1, *_ = lstsq(X1, signal, rcond=None)
    residual1 = signal - X1 @ coeffs1
    rss_1mode = float(np.dot(residual1, residual1))

    if not math.isfinite(rss_1mode) or rss_1mode < 0.0:
        return _null_result(valid_1mode=False, valid_2mode=False, trace_extra={"rss_1mode_invalid": True})

    # BIC validity gate: n must exceed k + margin for BIC to be interpretable.
    valid_bic_1mode = n > k_1mode + _N_MIN_BIC_MARGIN
    rss_floored_1mode = rss_1mode < n * _RSS_FLOOR  # floor was/will be applied
    bic_1mode: float | None = (
        float(k_1mode * math.log(n) + n * math.log(max(rss_1mode / n, _RSS_FLOOR)))
        if valid_bic_1mode else None
    )
    trace_1mode: dict[str, Any] = {
        **trace_base,
        "ln_f220": ln_f220,
        "ln_Q220": ln_q220,
        "rss_floored_1mode": rss_floored_1mode,
        "valid_bic_1mode": valid_bic_1mode,
    }

    # --- Validate 221 point estimates ---
    valid_2mode = (
        ok_221
        and mode_221.get("ln_f") is not None
        and mode_221.get("ln_Q") is not None
        and math.isfinite(float(mode_221["ln_f"]))
        and math.isfinite(float(mode_221["ln_Q"]))
    )
    if not valid_2mode:
        return {
            "schema_version": "model_comparison_v1",
            "n_samples": n,
            "k_1mode": k_1mode,
            "k_2mode": k_2mode,
            "rss_1mode": rss_1mode,
            "rss_2mode": None,
            "bic_1mode": bic_1mode,
            "bic_2mode": None,
            "delta_bic": None,
            "thresholds": {"two_mode_preferred_delta_bic": delta_bic_threshold},
            "conventions": conventions,
            "decision": {"two_mode_preferred": None},
            "valid_1mode": True,
            "valid_2mode": False,
            "valid_bic_1mode": valid_bic_1mode,
            "valid_bic_2mode": False,
            "trace": trace_1mode,
        }

    ln_f221 = float(mode_221["ln_f"])
    ln_q221 = float(mode_221["ln_Q"])
    f221 = math.exp(ln_f221)
    q221 = math.exp(ln_q221)
    tau221 = q221 / (math.pi * f221)
    w221 = 2.0 * math.pi * f221
    env221 = np.exp(-t / tau221)
    col_221c = env221 * np.cos(w221 * t)
    col_221s = env221 * np.sin(w221 * t)
    X2 = np.vstack([col_220c, col_220s, col_221c, col_221s]).T

    coeffs2, *_ = lstsq(X2, signal, rcond=None)
    residual2 = signal - X2 @ coeffs2
    rss_2mode = float(np.dot(residual2, residual2))

    if not math.isfinite(rss_2mode) or rss_2mode < 0.0:
        return {
            "schema_version": "model_comparison_v1",
            "n_samples": n,
            "k_1mode": k_1mode,
            "k_2mode": k_2mode,
            "rss_1mode": rss_1mode,
            "rss_2mode": None,
            "bic_1mode": bic_1mode,
            "bic_2mode": None,
            "delta_bic": None,
            "thresholds": {"two_mode_preferred_delta_bic": delta_bic_threshold},
            "conventions": conventions,
            "decision": {"two_mode_preferred": None},
            "valid_1mode": True,
            "valid_2mode": False,
            "valid_bic_1mode": valid_bic_1mode,
            "valid_bic_2mode": False,
            "trace": {**trace_1mode, "rss_2mode_invalid": True},
        }

    # BIC validity gate for 2-mode.
    valid_bic_2mode = n > k_2mode + _N_MIN_BIC_MARGIN
    rss_floored_2mode = rss_2mode < n * _RSS_FLOOR

    bic_2mode: float | None
    delta_bic: float | None
    two_mode_preferred: bool | None
    if valid_bic_1mode and valid_bic_2mode:
        bic_2mode = float(k_2mode * math.log(n) + n * math.log(max(rss_2mode / n, _RSS_FLOOR)))
        delta_bic = bic_2mode - bic_1mode  # type: ignore[operator]  # bic_1mode is float here
        # Tie-breaking: near-zero delta_bic -> prefer simpler 1-mode deterministically.
        if abs(delta_bic) < _DELTA_BIC_TIE_EPS:
            delta_bic = 0.0
            two_mode_preferred = False
        else:
            two_mode_preferred = bool(delta_bic < delta_bic_threshold)
    else:
        bic_2mode = None
        delta_bic = None
        two_mode_preferred = None

    return {
        "schema_version": "model_comparison_v1",
        "n_samples": n,
        "k_1mode": k_1mode,
        "k_2mode": k_2mode,
        "rss_1mode": rss_1mode,
        "rss_2mode": rss_2mode,
        "bic_1mode": bic_1mode,
        "bic_2mode": bic_2mode,
        "delta_bic": delta_bic,
        "thresholds": {"two_mode_preferred_delta_bic": delta_bic_threshold},
        "conventions": conventions,
        "decision": {"two_mode_preferred": two_mode_preferred},
        "valid_1mode": True,
        "valid_2mode": True,
        "valid_bic_1mode": valid_bic_1mode,
        "valid_bic_2mode": valid_bic_2mode,
        "trace": {
            **trace_1mode,
            "ln_f221": ln_f221,
            "ln_Q221": ln_q221,
            "rss_floored_2mode": rss_floored_2mode,
            "valid_bic_2mode": valid_bic_2mode,
        },
    }


def build_results_payload(
    run_id: str,
    window_meta: dict[str, Any] | None,
    mode_220: dict[str, Any],
    mode_220_ok: bool,
    mode_221: dict[str, Any],
    mode_221_ok: bool,
    flags: list[str],
    *,
    model_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verdict = "OK" if mode_220_ok else "INSUFFICIENT_DATA"
    messages: list[str] = []
    if mode_220_ok and not mode_221_ok:
        messages.append("Mode 221 insufficient/unstable; proceeding with 220-only result.")
    if verdict == "INSUFFICIENT_DATA":
        messages.append("Best-effort multimode fit: one or more modes unavailable or unstable.")

    results: dict[str, Any] = {
        "verdict": verdict,
        "quality_flags": sorted(set(flags)),
        "messages": sorted(messages),
    }
    if model_comparison is not None:
        two_mode_preferred = bool(
            (model_comparison.get("decision") or {}).get("two_mode_preferred", False)
        )
        results["quality_gates"] = {"two_mode_preferred": two_mode_preferred}

    return {
        "schema_version": "multimode_estimates_v1",
        "run_id": run_id,
        "source": {"stage": "s2_ringdown_window", "window": window_meta},
        "modes_target": TARGET_MODES,
        "results": results,
        "modes": [mode_220, mode_221],
    }


def _json_strictify(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_strictify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_strictify(v) for v in value]
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s3b multimode estimates")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--s3-estimates", default=None, help="Path to s3_ringdown_estimates outputs/estimates.json")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--method",
        default="hilbert_peakband",
        choices=["hilbert_peakband", "spectral_two_pass"],
    )
    ap.add_argument("--max-lnf-span-220", type=float, default=1.0)
    ap.add_argument("--max-lnq-span-220", type=float, default=3.0)
    ap.add_argument("--max-lnf-span-221", type=float, default=1.0)
    ap.add_argument("--max-lnq-span-221", type=float, default=1.0)
    ap.add_argument("--min-valid-fraction-221", type=float, default=0.8)
    ap.add_argument(
        "--cv-threshold-221",
        type=float,
        default=1.0,
        help="Only adds flags; does NOT gate verdict directly",
    )
    args = ap.parse_args()

    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).resolve())

    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "method": args.method,
            "max_lnf_span_220": args.max_lnf_span_220,
            "max_lnq_span_220": args.max_lnq_span_220,
            "max_lnf_span_221": args.max_lnf_span_221,
            "max_lnq_span_221": args.max_lnq_span_221,
            "min_valid_fraction_221": args.min_valid_fraction_221,
            "cv_threshold_221": args.cv_threshold_221,
        },
    )

    try:
        window_meta: dict[str, Any] | None = None
        window_meta_path = _discover_s2_window_meta(ctx.run_dir)
        if window_meta_path is not None and window_meta_path.exists():
            window_meta = json.loads(window_meta_path.read_text(encoding="utf-8"))

        s3_estimates_path = _resolve_s3_estimates(ctx.run_dir, args.s3_estimates)
        band_low, band_high = _load_s3_band(s3_estimates_path, window_meta=window_meta)
        global_flags: list[str] = []
        if window_meta is None:
            global_flags.append("missing_window_meta")
            window_meta_path = ctx.run_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"

        npz_path = _discover_s2_npz(ctx.run_dir)
        check_inputs(
            ctx,
            {"s2_rd_npz": npz_path, "s3_estimates": s3_estimates_path},
            optional={"s2_window_meta": window_meta_path},
        )
        signal, fs = _load_signal_from_npz(npz_path, window_meta=window_meta)

        if args.method == "spectral_two_pass":
            est_220 = lambda sig, sr: _estimate_220_spectral(sig, sr, band_low=band_low, band_high=band_high)
            est_221 = lambda sig, sr: _estimate_221_spectral_two_pass(sig, sr, band_low=band_low, band_high=band_high)
        else:
            est_220 = lambda sig, sr: _estimate_220(sig, sr, band_low=band_low, band_high=band_high)
            est_221 = lambda sig, sr: _estimate_221_from_signal(sig, sr, band_low=band_low, band_high=band_high)

        mode_220, flags_220, ok_220 = evaluate_mode(
            signal,
            fs,
            label="220",
            mode=[2, 2, 0],
            estimator=est_220,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            max_lnf_span=args.max_lnf_span_220,
            max_lnq_span=args.max_lnq_span_220,
            min_point_samples=50,
            min_point_valid_fraction=0.5,
            method=args.method,
        )
        mode_221, flags_221, ok_221 = evaluate_mode(
            signal,
            fs,
            label="221",
            mode=[2, 2, 1],
            estimator=est_221,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 1,
            min_valid_fraction=args.min_valid_fraction_221,
            cv_threshold=args.cv_threshold_221,
            max_lnf_span=args.max_lnf_span_221,
            max_lnq_span=args.max_lnq_span_221,
            min_point_samples=50,
            min_point_valid_fraction=0.5,
            method=args.method,
        )

        model_comparison = compute_model_comparison(
            signal, fs, mode_220, mode_221, ok_220, ok_221,
        )

        payload = build_results_payload(
            args.run_id,
            window_meta,
            mode_220,
            ok_220,
            mode_221,
            ok_221,
            global_flags + flags_220 + flags_221,
            model_comparison=model_comparison,
        )

        out_path = ctx.outputs_dir / "multimode_estimates.json"
        write_json_atomic(out_path, _json_strictify(payload))

        comparison_path = ctx.outputs_dir / "model_comparison.json"
        write_json_atomic(comparison_path, _json_strictify(model_comparison))

        finalize(
            ctx,
            {"multimode_estimates": out_path, "model_comparison": comparison_path},
            verdict="PASS",
            results=payload["results"],
        )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
