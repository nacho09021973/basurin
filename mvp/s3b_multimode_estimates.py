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
from mvp.multimode_viability import (
    DEFAULT_THRESHOLDS,
    classify_multimode_viability as classify_multimode_viability_v3,
    evaluate_science_evidence,
    evaluate_systematics_gate,
)
from mvp.gwtc_events import get_event
from mvp.kerr_qnm_fits import kerr_qnm
from mvp.s3_ringdown_estimates import (
    _compute_kerr_centered_band,
    _resolve_f220_kerr_hz,
    estimate_ringdown_observables,
    estimate_ringdown_spectral,
)

STAGE = "s3b_multimode_estimates"
TARGET_MODES = [
    {"mode": [2, 2, 0], "label": "220"},
    {"mode": [2, 2, 1], "label": "221"},
]
MIN_BOOTSTRAP_SAMPLES = 128
SIGMA_DET_EPS = 1e-12
SIGMA_COND_MAX = 1e12
MULTIMODE_OK = "MULTIMODE_OK"
SINGLEMODE_ONLY = "SINGLEMODE_ONLY"
RINGDOWN_NONINFORMATIVE = "RINGDOWN_NONINFORMATIVE"

# --- model_comparison numerical guards (documented in output conventions block) ---
_RSS_FLOOR = 1e-300         # floor applied to rss/n inside log; prevents -inf on perfect fits
_N_MIN_BIC_MARGIN = 2       # n must exceed k + this margin for BIC to be interpretable
_DELTA_BIC_TIE_EPS = 1e-10  # |delta_bic| < eps → prefer simpler 1-mode (deterministic tie-break)
S3B_220_BAND_WIDTH_FACTOR = 0.8
S3B_220_MIN_HALF_WIDTH_HZ = 15.0
S3B_220_Q_REF = 12.0
S3B_FREQ_FLOOR_HZ = 10.0
S3B_221_OVERLAP_LOW_PAD_FRAC = 0.1
S3B_221_OVERLAP_HIGH_PAD_FRAC = 0.2
MODE_221_RESIDUAL_STRATEGIES = ("refit_220_each_iter", "fixed_220_template")
MODE_221_TOPOLOGIES = ("rigid_spectral_split", "shared_band_early_taper")
MODE_221_EARLY_TAPER_TAU_FACTOR = 1.0


def _resolve_f221_kerr_hz(event_id: str) -> float | None:
    """Resolve Kerr f221 from catalog remnant mass/spin when available."""
    evt = get_event(event_id)
    if not evt:
        return None
    m_final = evt.get("m_final_msun")
    chi_final = evt.get("chi_final")
    if m_final is None or chi_final is None:
        return None
    qnm = kerr_qnm(float(m_final), float(chi_final), (2, 2, 1))
    if not math.isfinite(qnm.f_hz):
        return None
    return float(qnm.f_hz)


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


def _load_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid json payload in {path}")
    return payload


def _load_s3_band(payload: dict[str, Any], *, window_meta: dict[str, Any] | None) -> tuple[float, float]:
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


def _resolve_event_id(
    *,
    window_meta: dict[str, Any] | None,
    s3_payload: dict[str, Any] | None,
) -> str | None:
    for source in (window_meta, s3_payload):
        if not isinstance(source, dict):
            continue
        raw = source.get("event_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _load_s3_selected_t0_offset_ms(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("t0_selected")
    if not isinstance(selected, dict):
        return 0.0
    value = selected.get("offset_ms")
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        off = float(value)
        if math.isfinite(off) and off >= 0.0:
            return off
    return 0.0


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




def _load_measured_psd(psd_path: Path) -> dict[str, Any]:
    payload = json.loads(psd_path.read_text(encoding="utf-8"))
    freqs = np.asarray(payload.get("frequencies_hz", []), dtype=np.float64)
    if freqs.ndim != 1 or freqs.size < 2 or not np.all(np.isfinite(freqs)):
        raise RuntimeError(f"invalid measured PSD frequencies_hz: {psd_path}")
    if not np.all(np.diff(freqs) > 0):
        raise RuntimeError(f"invalid measured PSD frequencies_hz (non-monotonic): {psd_path}")
    return payload


def _detector_from_npz_path(npz_path: Path) -> str | None:
    stem = npz_path.stem.upper()
    for det in ("H1", "L1", "V1"):
        if det in stem:
            return det
    return None


def _whiten_with_measured_psd(
    signal: np.ndarray,
    fs: float,
    *,
    detector: str,
    psd_payload: dict[str, Any],
) -> np.ndarray:
    det_key = f"psd_{detector.upper()}"
    raw = psd_payload.get(det_key)
    if not isinstance(raw, list):
        raise KeyError(det_key)

    psd_freqs = np.asarray(psd_payload.get("frequencies_hz", []), dtype=float)
    psd_vals = np.asarray(raw, dtype=float)
    if psd_vals.shape != psd_freqs.shape:
        raise RuntimeError(f"{det_key} length mismatch with frequencies_hz")
    if np.any(~np.isfinite(psd_vals)):
        raise RuntimeError(f"{det_key} contains non-finite values")
    psd_vals = np.maximum(psd_vals, 1e-30)

    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    interp = np.interp(freqs, psd_freqs, psd_vals, left=psd_vals[0], right=psd_vals[-1])
    interp = np.maximum(interp, 1e-30)

    spec = np.fft.rfft(signal)
    white_spec = spec / np.sqrt(interp)
    return np.fft.irfft(white_spec, n=signal.size)


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


def _estimate_observables_in_band(signal: np.ndarray, fs: float, *, band: tuple[float, float]) -> dict[str, float]:
    return estimate_ringdown_observables(signal, fs, band[0], band[1])


def _estimate_spectral_in_band(signal: np.ndarray, fs: float, *, band: tuple[float, float]) -> dict[str, float]:
    est = estimate_ringdown_spectral(signal, fs, band)
    return {
        "f_hz": float(est["f_hz"]),
        "Q": float(est["Q"]),
        "tau_s": float(est["tau_s"]),
    }


def _split_mode_bands(*, band_low: float, band_high: float) -> tuple[tuple[float, float], tuple[float, float]]:
    # Split at 60% to keep mode-220 (fundamental) well inside its sub-band.
    # QNM ratio f_221/f_220 ≈ 1.5–1.7, so f_220 sits in the lower ~40% of
    # a typical [150,400] Hz band.  A 50/50 split places the boundary at
    # 275 Hz — only ~24 Hz above f_220 ≈ 251 Hz for GW150914, which trips
    # the spectral estimator's df_floor band-edge guard (≈25 Hz for a 40 ms
    # window) and forces a biased Hilbert fallback.
    mid = band_low + 0.6 * (band_high - band_low)
    eps = 1e-6
    band_220 = (band_low, max(mid - eps, band_low + eps))
    band_221 = (min(mid + eps, band_high - eps), band_high)
    return band_220, band_221


def _resolve_mode_bands(
    *,
    band_low: float,
    band_high: float,
    event_id: str | None,
    band_strategy: str = "kerr_centered_overlap",
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    fallback_220, fallback_221 = _split_mode_bands(band_low=band_low, band_high=band_high)
    default_strategy: dict[str, Any] = {
        "method": "default_split_60_40",
        "event_id": event_id,
        "mode_220_band_hz": [float(fallback_220[0]), float(fallback_220[1])],
        "mode_221_band_hz": [float(fallback_221[0]), float(fallback_221[1])],
    }
    if band_strategy == "default_split_60_40":
        return fallback_220, fallback_221, default_strategy

    if not event_id:
        return fallback_220, fallback_221, default_strategy

    f220_kerr_hz = _resolve_f220_kerr_hz(event_id)
    if f220_kerr_hz is None or not math.isfinite(float(f220_kerr_hz)):
        return fallback_220, fallback_221, default_strategy

    kerr_low, kerr_high, half_width_hz = _compute_kerr_centered_band(
        f220_kerr_hz=float(f220_kerr_hz),
        width_factor=S3B_220_BAND_WIDTH_FACTOR,
        min_half_width_hz=S3B_220_MIN_HALF_WIDTH_HZ,
        q_ref=S3B_220_Q_REF,
        f_min_floor_hz=S3B_FREQ_FLOOR_HZ,
    )
    eps = 1e-6
    band_220 = (max(float(band_low), float(kerr_low)), min(float(band_high), float(kerr_high)))
    if not band_220[1] > band_220[0] + eps:
        return fallback_220, fallback_221, default_strategy

    f221_kerr_hz = _resolve_f221_kerr_hz(event_id)
    if f221_kerr_hz is None or not math.isfinite(float(f221_kerr_hz)):
        return fallback_220, fallback_221, default_strategy

    raw_221 = (
        max(float(band_low), float(f221_kerr_hz) - half_width_hz),
        min(float(band_high), float(f221_kerr_hz) + half_width_hz),
    )
    if not raw_221[1] > raw_221[0] + eps:
        return fallback_220, fallback_221, default_strategy

    pad_low_hz = float(S3B_221_OVERLAP_LOW_PAD_FRAC) * float(half_width_hz)
    pad_high_hz = float(S3B_221_OVERLAP_HIGH_PAD_FRAC) * float(half_width_hz)
    band_221 = (
        max(float(band_low), float(raw_221[0]) - pad_low_hz),
        min(float(band_high), float(raw_221[1]) + pad_high_hz),
    )
    if not band_221[1] > band_221[0] + eps:
        return fallback_220, fallback_221, default_strategy

    if band_strategy == "coherent_harmonic_band":
        shared_band = _mode_221_shared_band(band_220, band_221)
        strategy = {
            "method": "coherent_harmonic_band",
            "event_id": event_id,
            "shared_band_hz": [float(shared_band[0]), float(shared_band[1])],
            "shared_band_modes": ["220", "221"],
            "f220_kerr_hz": float(f220_kerr_hz),
            "f221_kerr_hz": float(f221_kerr_hz),
            "width_factor": float(S3B_220_BAND_WIDTH_FACTOR),
            "min_half_width_hz": float(S3B_220_MIN_HALF_WIDTH_HZ),
            "half_width_hz": float(half_width_hz),
            "mode_221_low_pad_hz": float(pad_low_hz),
            "mode_221_high_pad_hz": float(pad_high_hz),
            "mode_220_band_hz": [float(band_220[0]), float(band_220[1])],
            "mode_221_band_hz": [float(band_220[0]), float(band_221[1])],
            "mode_220_band_role": "estimation_band",
            "mode_221_band_role": "shared_coherent_band",
            "logical_subbands_hz": {
                "220": [float(band_220[0]), float(band_220[1])],
                "221": [float(band_221[0]), float(band_221[1])],
            },
            "logical_subbands_within_shared_band": True,
        }
        return band_220, shared_band, strategy

    strategy = {
        "method": "kerr_centered_overlap",
        "event_id": event_id,
        "f220_kerr_hz": float(f220_kerr_hz),
        "f221_kerr_hz": float(f221_kerr_hz),
        "width_factor": float(S3B_220_BAND_WIDTH_FACTOR),
        "min_half_width_hz": float(S3B_220_MIN_HALF_WIDTH_HZ),
        "half_width_hz": float(half_width_hz),
        "mode_221_low_pad_hz": float(pad_low_hz),
        "mode_221_high_pad_hz": float(pad_high_hz),
        "mode_221_raw_band_hz": [float(raw_221[0]), float(raw_221[1])],
        "mode_220_band_hz": [float(band_220[0]), float(band_220[1])],
        "mode_221_band_hz": [float(band_221[0]), float(band_221[1])],
    }
    return band_220, band_221, strategy


def _estimate_220(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    band_220, _ = _split_mode_bands(band_low=band_low, band_high=band_high)
    return _estimate_observables_in_band(signal, fs, band=band_220)


def _estimate_220_spectral(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    band_220, _ = _split_mode_bands(band_low=band_low, band_high=band_high)
    return _estimate_spectral_in_band(signal, fs, band=band_220)


def _coerce_positive_finite_estimate(est: dict[str, float], key: str) -> float:
    value = est.get(key)
    coerced = float(value)
    if not math.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"invalid template estimate: {key}={value!r}")
    return coerced


def _template_220(signal: np.ndarray, fs: float, est220: dict[str, float]) -> np.ndarray:
    from numpy.linalg import lstsq

    t = np.arange(signal.size, dtype=float) / fs
    f = _coerce_positive_finite_estimate(est220, "f_hz")
    tau = _coerce_positive_finite_estimate(est220, "tau_s")
    env = np.exp(-t / tau)
    w = 2.0 * math.pi * f
    b1 = env * np.cos(w * t)
    b2 = env * np.sin(w * t)
    X = np.vstack([b1, b2]).T
    if not np.all(np.isfinite(X)):
        raise ValueError("invalid template estimate: non-finite 220 design matrix")
    coeffs, *_ = lstsq(X, signal, rcond=None)
    return (X @ coeffs).astype(float)


def _estimate_221_from_signal(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    est220 = _estimate_220(signal, fs, band_low=band_low, band_high=band_high)
    residual = signal - _template_220(signal, fs, est220)
    _, band_221 = _split_mode_bands(band_low=band_low, band_high=band_high)
    return _estimate_observables_in_band(residual, fs, band=band_221)


def _estimate_221_spectral_two_pass(signal: np.ndarray, fs: float, *, band_low: float, band_high: float) -> dict[str, float]:
    est220 = _estimate_220_spectral(signal, fs, band_low=band_low, band_high=band_high)
    residual = signal - _template_220(signal, fs, est220)
    _, band_221 = _split_mode_bands(band_low=band_low, band_high=band_high)
    return _estimate_spectral_in_band(residual, fs, band=band_221)


def _estimate_221_from_signal_in_bands(
    signal: np.ndarray,
    fs: float,
    *,
    band_220: tuple[float, float],
    band_221: tuple[float, float],
) -> dict[str, float]:
    est220 = _estimate_observables_in_band(signal, fs, band=band_220)
    residual = signal - _template_220(signal, fs, est220)
    return _estimate_observables_in_band(residual, fs, band=band_221)


def _estimate_221_spectral_two_pass_in_bands(
    signal: np.ndarray,
    fs: float,
    *,
    band_220: tuple[float, float],
    band_221: tuple[float, float],
) -> dict[str, float]:
    est220 = _estimate_spectral_in_band(signal, fs, band=band_220)
    residual = signal - _template_220(signal, fs, est220)
    return _estimate_spectral_in_band(residual, fs, band=band_221)


def _mode_221_shared_band(band_220: tuple[float, float], band_221: tuple[float, float]) -> tuple[float, float]:
    return float(band_220[0]), float(band_221[1])


def _apply_early_time_taper(signal: np.ndarray, fs: float, *, tau_s: float, tau_factor: float = MODE_221_EARLY_TAPER_TAU_FACTOR) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError("signal must be 1D for early-time taper")
    tau_eff = float(tau_s) * float(tau_factor)
    if not math.isfinite(tau_eff) or tau_eff <= 0.0:
        raise ValueError(f"invalid early-time taper tau: {tau_s!r}")
    t = np.arange(signal.size, dtype=float) / fs
    weights = np.exp(-t / tau_eff)
    return signal * weights


def _estimate_221_in_topology(
    signal: np.ndarray,
    fs: float,
    *,
    band_220: tuple[float, float],
    band_221: tuple[float, float],
    method: str,
    topology: str,
) -> dict[str, float]:
    estimator = _estimate_spectral_in_band if method == "spectral_two_pass" else _estimate_observables_in_band
    est220 = estimator(signal, fs, band=band_220)
    residual = signal - _template_220(signal, fs, est220)
    estimate_band = band_221
    if topology == "shared_band_early_taper":
        residual = _apply_early_time_taper(residual, fs, tau_s=float(est220["tau_s"]))
        estimate_band = _mode_221_shared_band(band_220, band_221)
    return estimator(residual, fs, band=estimate_band)


def _mode_221_residual_band_estimator(
    method: str,
    *,
    band_220: tuple[float, float],
    band_221: tuple[float, float],
    topology: str,
) -> Callable[[np.ndarray, float], dict[str, float]]:
    estimate_band = band_221 if topology == "rigid_spectral_split" else _mode_221_shared_band(band_220, band_221)
    if method == "spectral_two_pass":
        base_estimator = lambda sig, sr: _estimate_spectral_in_band(sig, sr, band=estimate_band)
    else:
        base_estimator = lambda sig, sr: _estimate_observables_in_band(sig, sr, band=estimate_band)
    return base_estimator


def _mode_221_refit_each_iter_estimator(
    method: str,
    *,
    band_220: tuple[float, float],
    band_221: tuple[float, float],
    topology: str,
) -> Callable[[np.ndarray, float], dict[str, float]]:
    return lambda sig, sr: _estimate_221_in_topology(
        sig,
        sr,
        band_220=band_220,
        band_221=band_221,
        method=method,
        topology=topology,
    )


def _prepare_mode_221_bootstrap_inputs(
    signal: np.ndarray,
    fs: float,
    *,
    method: str,
    band_220: tuple[float, float],
    band_221: tuple[float, float],
    residual_strategy: str,
    topology: str,
) -> tuple[np.ndarray, Callable[[np.ndarray, float], dict[str, float]], dict[str, Any], list[str]]:
    strategy_meta: dict[str, Any] = {
        "strategy": residual_strategy,
        "topology": topology,
        "template_scope": "per_bootstrap_iter",
        "bootstrap_signal": "full_signal",
    }
    if topology == "shared_band_early_taper":
        strategy_meta.update(
            {
                "estimation_band": "shared_220_221_band",
                "time_weighting": "early_exponential_taper",
                "time_weighting_tau_factor": float(MODE_221_EARLY_TAPER_TAU_FACTOR),
            }
        )
    else:
        strategy_meta["estimation_band"] = "rigid_221_band"
    if residual_strategy == "refit_220_each_iter":
        return (
            signal,
            _mode_221_refit_each_iter_estimator(
                method,
                band_220=band_220,
                band_221=band_221,
                topology=topology,
            ),
            strategy_meta,
            [],
        )

    if method == "spectral_two_pass":
        est220 = _estimate_spectral_in_band(signal, fs, band=band_220)
    else:
        est220 = _estimate_observables_in_band(signal, fs, band=band_220)
    residual_signal = signal - _template_220(signal, fs, est220)
    if topology == "shared_band_early_taper":
        residual_signal = _apply_early_time_taper(residual_signal, fs, tau_s=float(est220["tau_s"]))
    strategy_meta.update(
        {
            "template_scope": "full_signal_once",
            "bootstrap_signal": "fixed_220_residual",
        }
    )
    return (
        residual_signal,
        _mode_221_residual_band_estimator(
            method,
            band_220=band_220,
            band_221=band_221,
            topology=topology,
        ),
        strategy_meta,
        [],
    )


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

    # evaluate_mode operates on bootstrap samples in (ln f, ln Q) coordinates.
    # spectral_two_pass estimators already report Q, so applying an additional
    # tau->Q conversion here would inflate Q by a factor of pi*f.

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


def _mode_221_usable_reason(mode_221_ok: bool, flags: list[str]) -> str:
    """Determine the canonical reason for mode 221 usability.

    Returns ``"ok"`` when usable; otherwise the most relevant flag or reason.
    """
    if mode_221_ok:
        return "ok"
    # Prefer the first 221-specific quality flag as the reason.
    for flag in sorted(flags):
        if flag.startswith("221_"):
            return flag
    if flags:
        return flags[0]
    return "mode_221_not_ok"


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
    t0_offset_ms_from_s3: float = 0.0,
    band_strategy: dict[str, Any] | None = None,
    mode_221_residual: dict[str, Any] | None = None,
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

    source: dict[str, Any] = {"stage": "s2_ringdown_window", "window": window_meta}
    if t0_offset_ms_from_s3 > 0.0:
        source["t0_offset_ms_from_s3"] = float(t0_offset_ms_from_s3)
    if band_strategy is not None:
        source["band_strategy"] = band_strategy
    if mode_221_residual is not None:
        source["mode_221_residual"] = dict(mode_221_residual)
        source["mode_221_strategy"] = mode_221_residual.get("strategy")

    return {
        "schema_version": "multimode_estimates_v1",
        "run_id": run_id,
        "source": source,
        "modes_target": TARGET_MODES,
        "results": results,
        "modes": [mode_220, mode_221],
        "mode_221_usable": bool(mode_221_ok),
        "mode_221_usable_reason": _mode_221_usable_reason(mode_221_ok, flags),
    }


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _load_optional_multimode_inputs(run_dir: Path) -> dict[str, Any]:
    optional = run_dir / "external_inputs" / "multimode_viability_inputs.json"
    if not optional.exists():
        return {}
    try:
        payload = json.loads(optional.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _mode_frequency_summary(mode_payload: dict[str, Any]) -> dict[str, float | None]:
    fit = mode_payload.get("fit") if isinstance(mode_payload, dict) else None
    stability = fit.get("stability") if isinstance(fit, dict) else {}

    lnf_p10 = _as_float_or_none(stability.get("lnf_p10"))
    lnf_p50 = _as_float_or_none(stability.get("lnf_p50"))
    lnf_p90 = _as_float_or_none(stability.get("lnf_p90"))

    f_p10 = math.exp(lnf_p10) if lnf_p10 is not None else None
    f_p50 = math.exp(lnf_p50) if lnf_p50 is not None else None
    f_p90 = math.exp(lnf_p90) if lnf_p90 is not None else None

    return {
        "f_p10": f_p10,
        "f_median": f_p50,
        "f_p90": f_p90,
        "f_iqr": (f_p90 - f_p10) if (f_p10 is not None and f_p90 is not None) else None,
    }


def _mode_q_median(mode_payload: dict[str, Any]) -> float | None:
    fit = mode_payload.get("fit") if isinstance(mode_payload, dict) else None
    stability = fit.get("stability") if isinstance(fit, dict) else {}
    lnq_p50 = _as_float_or_none(stability.get("lnQ_p50"))
    if lnq_p50 is not None:
        return math.exp(lnq_p50)
    lnq = _as_float_or_none(mode_payload.get("ln_Q"))
    return math.exp(lnq) if lnq is not None else None


def _rf_quantiles_from_modes(mode_220: dict[str, Any], mode_221: dict[str, Any]) -> dict[str, float] | None:
    f220 = _mode_frequency_summary(mode_220)
    f221 = _mode_frequency_summary(mode_221)
    values = (f220["f_p10"], f220["f_median"], f220["f_p90"], f221["f_p10"], f221["f_median"], f221["f_p90"])
    if not all(_as_float_or_none(v) is not None and float(v) > 0.0 for v in values):
        return None

    q05 = float(f221["f_p10"] / f220["f_p90"])
    q50 = float(f221["f_median"] / f220["f_median"])
    q95 = float(f221["f_p90"] / f220["f_p10"])
    if not (math.isfinite(q05) and math.isfinite(q50) and math.isfinite(q95) and q05 <= q50 <= q95):
        return None
    return {"q05": q05, "q50": q50, "q95": q95}


def classify_multimode_viability(
    *,
    boundary_fraction: float | None,
    valid_fraction_220: float | None,
    valid_fraction_221: float | None,
    boundary_fraction_threshold: float = 0.95,
    valid_fraction_floor: float = 0.5,
) -> dict[str, Any]:
    """Compat wrapper (legacy tests) backed by multimode_viability_v3."""
    if valid_fraction_220 is not None and float(valid_fraction_220) < float(valid_fraction_floor):
        return {
            "class": RINGDOWN_NONINFORMATIVE,
            "reasons": ["VALID_FRACTION_220_LOW"],
            "metrics": {
                "boundary_fraction": boundary_fraction,
                "valid_fraction": {"220": valid_fraction_220, "221": valid_fraction_221},
            },
        }
    if boundary_fraction is not None and float(boundary_fraction) >= float(boundary_fraction_threshold):
        return {
            "class": SINGLEMODE_ONLY,
            "reasons": ["BOUNDARY_FRACTION_HIGH"],
            "metrics": {
                "boundary_fraction": boundary_fraction,
                "valid_fraction": {"220": valid_fraction_220, "221": valid_fraction_221},
            },
        }

    return classify_multimode_viability_v3(
        {
            "valid_fraction_220": valid_fraction_220,
            "valid_fraction_221": valid_fraction_221,
            "mode_221_ok": True,
            "f_220_median": 1.0,
            "f_220_iqr": 0.0,
            "f_221_median": 1.0,
            "f_221_iqr": 0.0,
            "Rf_bootstrap_quantiles": {"q05": 0.9, "q50": 1.0, "q95": 1.1},
            "Rf_kerr_band": [0.8, 1.2],
            "delta_bic": DEFAULT_THRESHOLDS["DELTA_BIC_SUPPORTIVE"],
            "two_mode_preferred": True,
        }
    )


def _compute_multimode_summary_blocks(
    mode_220: dict[str, Any],
    mode_221: dict[str, Any],
    mode_221_ok: bool,
    model_comparison: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    optional_inputs = _load_optional_multimode_inputs(run_dir)

    m220 = _mode_frequency_summary(mode_220)
    m221 = _mode_frequency_summary(mode_221)
    rf_quantiles = _rf_quantiles_from_modes(mode_220, mode_221)

    viability_inputs = {
        "valid_fraction_220": mode_220.get("fit", {}).get("stability", {}).get("valid_fraction"),
        "valid_fraction_221": mode_221.get("fit", {}).get("stability", {}).get("valid_fraction"),
        "mode_221_ok": bool(mode_221_ok),
        "f_220_median": m220["f_median"],
        "f_220_iqr": m220["f_iqr"],
        "f_221_median": m221["f_median"],
        "f_221_iqr": m221["f_iqr"],
        "Q_221_median": _mode_q_median(mode_221),
        "Rf_bootstrap_quantiles": rf_quantiles,
        "Rf_kerr_band": optional_inputs.get("Rf_kerr_band"),
        "delta_bic": model_comparison.get("delta_bic"),
        "two_mode_preferred": (model_comparison.get("decision") or {}).get("two_mode_preferred"),
    }
    viability = classify_multimode_viability_v3(viability_inputs)

    systematics_gate = evaluate_systematics_gate(
        {
            "t0_plateau": optional_inputs.get("t0_plateau"),
            "chi_psd_at_f221": optional_inputs.get("chi_psd_at_f221"),
            "Q_221_median": viability_inputs["Q_221_median"],
        }
    )

    science_evidence = evaluate_science_evidence(
        viability=viability,
        systematics=systematics_gate,
        rf_bootstrap_quantiles=rf_quantiles,
        rf_kerr_grid=optional_inputs.get("rf_kerr_grid") or [],
        chi_grid=optional_inputs.get("chi_grid") or [],
        override=optional_inputs.get("systematics_override"),
    )

    annotations = {
        "kerr_inconsistency_is_not_fail": True,
        "rf_quantiles_source": "derived_from_mode_bootstrap_percentiles" if rf_quantiles else "not_available",
    }
    return viability, systematics_gate, science_evidence, annotations


def _json_strictify(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_strictify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_strictify(v) for v in value]
    return value


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MVP s3b multimode estimates")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--s3-estimates", default=None, help="Path to s3_ringdown_estimates outputs/estimates.json")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--psd-path", default=None, help="Path to measured_psd.json from extract_psd.py")
    ap.add_argument(
        "--method",
        default="hilbert_peakband",
        choices=["hilbert_peakband", "spectral_two_pass"],
    )
    ap.add_argument(
        "--bootstrap-221-residual-strategy",
        default="refit_220_each_iter",
        choices=list(MODE_221_RESIDUAL_STRATEGIES),
    )
    ap.add_argument(
        "--mode-221-topology",
        default="rigid_spectral_split",
        choices=list(MODE_221_TOPOLOGIES),
        help="Opt-in 221 extraction topology; default preserves the current rigid spectral split.",
    )
    ap.add_argument(
        "--band-strategy",
        default="kerr_centered_overlap",
        choices=["default_split_60_40", "kerr_centered_overlap", "coherent_harmonic_band"],
        help="Band allocation strategy for the canonical 220/221 multimode stage.",
    )
    ap.add_argument("--max-lnf-span-220", type=float, default=1.0)
    ap.add_argument("--max-lnq-span-220", type=float, default=3.0)
    ap.add_argument("--max-lnf-span-221", type=float, default=1.0)
    ap.add_argument("--max-lnq-span-221", type=float, default=1.0)
    ap.add_argument("--min-valid-fraction-221", type=float, default=0.5)
    ap.add_argument(
        "--cv-threshold-221",
        type=float,
        default=1.0,
        help="Only adds flags; does NOT gate verdict directly",
    )
    return ap


def main() -> int:
    ap = _build_arg_parser()
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
            "bootstrap_221_residual_strategy": args.bootstrap_221_residual_strategy,
            "mode_221_topology": args.mode_221_topology,
            "band_strategy": args.band_strategy,
            "psd_path": args.psd_path,
            "max_lnf_span_220": args.max_lnf_span_220,
            "max_lnq_span_220": args.max_lnq_span_220,
            "max_lnf_span_221": args.max_lnf_span_221,
            "max_lnq_span_221": args.max_lnq_span_221,
            "min_valid_fraction_221": args.min_valid_fraction_221,
            "cv_threshold_221": args.cv_threshold_221,
            "psd_path": args.psd_path,
        },
    )

    try:
        window_meta: dict[str, Any] | None = None
        window_meta_path = _discover_s2_window_meta(ctx.run_dir)
        if window_meta_path is not None and window_meta_path.exists():
            window_meta = json.loads(window_meta_path.read_text(encoding="utf-8"))

        s3_estimates_path = _resolve_s3_estimates(ctx.run_dir, args.s3_estimates)
        s3_payload = _load_json_payload(s3_estimates_path)
        band_low, band_high = _load_s3_band(s3_payload, window_meta=window_meta)
        event_id = _resolve_event_id(window_meta=window_meta, s3_payload=s3_payload)
        band_220, band_221, band_strategy = _resolve_mode_bands(
            band_low=band_low,
            band_high=band_high,
            event_id=event_id,
            band_strategy=args.band_strategy,
        )
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
        psd_source = "internal_welch"
        if args.psd_path:
            try:
                psd_payload = _load_measured_psd(Path(args.psd_path))
                detector = _detector_from_npz_path(npz_path)
                if detector is None:
                    print(f"[{STAGE}] WARNING: unable to infer detector from {npz_path.name}; falling back to internal Welch", flush=True)
                else:
                    signal = _whiten_with_measured_psd(signal, fs, detector=detector, psd_payload=psd_payload)
                    psd_source = "external_measured_psd"
            except KeyError as exc:
                print(f"[{STAGE}] WARNING: measured PSD missing detector for {npz_path.name}: {exc}; falling back to internal Welch", flush=True)

        selected_t0_offset_ms = _load_s3_selected_t0_offset_ms(s3_estimates_path)
        applied_t0_offset_ms = 0.0
        if selected_t0_offset_ms > 0.0:
            shift = int(round((selected_t0_offset_ms / 1000.0) * fs))
            if 0 < shift < signal.size - 16:
                signal = signal[shift:]
                applied_t0_offset_ms = selected_t0_offset_ms
            else:
                global_flags.append("s3_t0_selected_offset_out_of_range")

        if args.method == "spectral_two_pass":
            est_220 = lambda sig, sr: _estimate_spectral_in_band(sig, sr, band=band_220)
        else:
            est_220 = lambda sig, sr: _estimate_observables_in_band(sig, sr, band=band_220)
        mode_221_signal, est_221, mode_221_residual, mode_221_strategy_flags = _prepare_mode_221_bootstrap_inputs(
            signal,
            fs,
            method=args.method,
            band_220=band_220,
            band_221=band_221,
            residual_strategy=args.bootstrap_221_residual_strategy,
            topology=args.mode_221_topology,
        )
        global_flags.extend(mode_221_strategy_flags)

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
            mode_221_signal,
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
            t0_offset_ms_from_s3=applied_t0_offset_ms,
            band_strategy=band_strategy,
            mode_221_residual=mode_221_residual,
        )

        out_path = ctx.outputs_dir / "multimode_estimates.json"
        write_json_atomic(out_path, _json_strictify(payload))

        comparison_path = ctx.outputs_dir / "model_comparison.json"
        write_json_atomic(comparison_path, _json_strictify(model_comparison))

        viability, systematics_gate, science_evidence, annotations = _compute_multimode_summary_blocks(
            mode_220=mode_220,
            mode_221=mode_221,
            mode_221_ok=ok_221,
            model_comparison=model_comparison,
            run_dir=ctx.run_dir,
        )

        finalize(
            ctx,
            {"multimode_estimates": out_path, "model_comparison": comparison_path},
            verdict="PASS",
            results=payload["results"],
            extra_summary={
                "multimode_viability": viability,
                "systematics_gate": systematics_gate,
                "science_evidence": science_evidence,
                "annotations": annotations,
                "psd_source": psd_source,
            },
        )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
