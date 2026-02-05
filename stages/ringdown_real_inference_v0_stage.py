#!/usr/bin/env python3
"""
stages/ringdown_real_inference_v0_stage.py
---------------------------------------
Canonical stage: minimal physical inference on real ringdown window.

QNM damped sinusoid fit (v1):
  h(t) = A * exp(-(t-t0)/tau) * cos(2*pi*f*(t-t0) + phi) + C
  Fitted via scipy.optimize.least_squares with analytical Jacobian.
  Produces qnm_fit block in inference_report.json and decision_qnm verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1], _here.parents[2]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2
STAGE_NAME_DEFAULT = "ringdown_real_inference_v0"
STRAIN_KEYS = ["strain", "h", "data", "x", "rd_strain"]
FS_KEYS = ["sample_rate_hz", "fs_hz", "fs", "sample_rate", "sr"]
MIN_BLOCKS_TAU = 5
TAU_FRAC_DIFF_MAX = 0.2
DEFAULT_STABILITY_K = 3.0
DEFAULT_LOO_Z_THRESHOLD = 3.0
DEFAULT_CLIPPING_FRACTION_MAX = 0.05


def _parse_band_hz(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("band-hz debe tener formato 'low,high'")
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("band-hz debe ser numérico") from exc


def _abort(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_single_jsonl(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path} debe tener exactamente 1 línea (tiene {len(lines)})")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido en {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} debe contener un objeto JSON")
    return payload


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} no es finito: {value}")
    return value


def _require_positive(value: float, label: str) -> float:
    value = _require_finite(value, label)
    if value <= 0:
        raise ValueError(f"{label} debe ser > 0 (got {value})")
    return value


def _load_strain_and_fs(
    path: Path, fallback_fs: float | None
) -> tuple[np.ndarray, float, str]:
    try:
        data = np.load(path)
    except Exception as exc:
        raise RuntimeError(f"no se pudo leer npz {path}: {exc}") from exc

    strain = None
    for key in STRAIN_KEYS:
        if key in data:
            strain = np.asarray(data[key], dtype=float)
            break
    if strain is None:
        available = ", ".join(list(data.files))
        raise RuntimeError(
            f"strain no encontrado en {path}; claves disponibles: [{available}]"
        )
    if strain.ndim != 1:
        raise RuntimeError(f"strain debe ser 1D en {path}")
    if not np.all(np.isfinite(strain)):
        raise RuntimeError(f"strain contiene NaN/Inf en {path}")

    fs = None
    fs_source = "fallback"
    for key in FS_KEYS:
        if key in data:
            fs = float(np.asarray(data[key]).reshape(-1)[0])
            fs_source = "npz"
            break
    if fs is None:
        fs = fallback_fs
    if fs is None:
        raise RuntimeError(f"fs_hz no encontrado en {path} ni en observables/features")
    fs = _require_positive(float(fs), f"fs_hz ({path})")

    return strain, fs, fs_source


def _estimate_f_peak(
    strain: np.ndarray, fs_hz: float, band_hz: list[float]
) -> tuple[float, float, float]:
    n = int(strain.size)
    if n <= 0:
        raise RuntimeError("strain vacío")
    window = np.hanning(n)
    spectrum = np.fft.rfft(strain * window)
    if not np.all(np.isfinite(spectrum)):
        raise RuntimeError("FFT contiene NaN/Inf")
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        raise RuntimeError(
            f"no hay bins en la banda {band_hz[0]}-{band_hz[1]} Hz"
        )

    band_freqs = freqs[mask]
    band_mag = mag[mask]
    idx = int(np.argmax(band_mag))
    f_peak = float(band_freqs[idx])
    peak_mag = float(band_mag[idx])
    df_hz = float(fs_hz / n)

    _require_finite(f_peak, "f_peak_hz")
    _require_finite(peak_mag, "peak_mag")
    _require_positive(df_hz, "df_hz")

    return f_peak, peak_mag, df_hz


def _estimate_tau(
    strain: np.ndarray, fs_hz: float,
) -> tuple[float | None, list[str], dict[str, Any]]:
    """Estimate exponential decay time.  Returns (tau_s, notes, metrics)."""
    notes: list[str] = []
    metrics: dict[str, Any] = {
        "block_len_s": None,
        "n_blocks_total": 0,
        "n_blocks_valid": 0,
        "reject_reasons": {},
    }
    n = int(strain.size)
    if n <= 0:
        notes.append("WARN: strain vacío para tau")
        return None, notes, metrics

    i0 = int(np.argmax(np.abs(strain)))
    block = max(16, int(0.01 * fs_hz))
    metrics["block_len_s"] = block / fs_hz

    amplitudes: list[float] = []
    times: list[float] = []
    start = i0
    while start < n:
        end = min(start + block, n)
        block_data = strain[start:end]
        if block_data.size == 0:
            break
        amp = float(np.sqrt(np.mean(np.square(block_data))))
        center = start + (block_data.size - 1) / 2.0
        t = (center - i0) / fs_hz
        amplitudes.append(amp)
        times.append(float(t))
        start = end

    metrics["n_blocks_total"] = len(amplitudes)

    if not amplitudes:
        notes.append("WARN: no se pudo construir envolvente para tau")
        return None, notes, metrics

    amp0 = amplitudes[0]
    if amp0 <= 0:
        notes.append("WARN: amplitude_0 no positiva para tau")
        return None, notes, metrics

    frac = 0.05
    reject_reasons: dict[str, int] = {}
    valid: list[tuple[float, float]] = []
    for t, a in zip(times, amplitudes):
        if a <= 0:
            reject_reasons["zero_amplitude"] = reject_reasons.get("zero_amplitude", 0) + 1
        elif a < frac * amp0:
            reject_reasons["low_power"] = reject_reasons.get("low_power", 0) + 1
        else:
            valid.append((t, a))

    metrics["n_blocks_valid"] = len(valid)
    metrics["reject_reasons"] = reject_reasons

    if len(valid) < MIN_BLOCKS_TAU:
        notes.append("WARN: menos de 5 bloques válidos para tau")
        return None, notes, metrics

    t_vals = np.array([item[0] for item in valid], dtype=float)
    a_vals = np.array([item[1] for item in valid], dtype=float)
    y = np.log(a_vals)
    A = np.vstack([np.ones_like(t_vals), t_vals]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    b = float(coeffs[1])

    if not math.isfinite(b):
        notes.append("WARN: ajuste lineal inválido para tau")
        return None, notes, metrics
    if b >= 0:
        notes.append("WARN: pendiente no negativa en ajuste de tau")
        return None, notes, metrics

    tau_s = -1.0 / b
    if not math.isfinite(tau_s) or tau_s <= 0:
        notes.append("WARN: tau no finito o no positivo")
        return None, notes, metrics

    return float(tau_s), notes, metrics


# ---------------------------------------------------------------------------
# QNM damped-sinusoid fit helpers (v1)
# ---------------------------------------------------------------------------
_SCIPY_AVAILABLE = False
try:
    from scipy.optimize import least_squares as _least_squares
    from scipy.signal import butter as _butter, sosfiltfilt as _sosfiltfilt
    from scipy.signal.windows import tukey as _tukey

    _SCIPY_AVAILABLE = True
except ImportError:
    pass

QNM_MODEL_VERSION = "damped_sinusoid_v1"
_QNM_MAX_NFEV = 2000
TAU_MIN_S = 0.01
TAU_MAX_S = 0.5
CLIP_EPS_FRAC = 0.01


def _is_clipped(x: float | None, lo: float, hi: float, eps_frac: float) -> bool | None:
    if x is None:
        return None
    eps = eps_frac * (hi - lo)
    return (x <= lo + eps) or (x >= hi - eps)


def _bandpass_filter(
    strain: np.ndarray, fs_hz: float, lo: float, hi: float,
) -> np.ndarray:
    """Bandpass filter.  Butterworth order 2 (scipy) or FFT hard filter (fallback)."""
    if _SCIPY_AVAILABLE:
        nyq = fs_hz / 2.0
        lo_n = max(lo / nyq, 1e-6)
        hi_n = min(hi / nyq, 1.0 - 1e-6)
        if lo_n >= hi_n:
            return strain.copy()
        sos = _butter(2, [lo_n, hi_n], btype="band", output="sos")
        return _sosfiltfilt(sos, strain).astype(float)
    # Fallback: FFT hard bandpass
    spectrum = np.fft.rfft(strain)
    freqs = np.fft.rfftfreq(len(strain), d=1.0 / fs_hz)
    mask = (freqs >= lo) & (freqs <= hi)
    spectrum[~mask] = 0.0
    return np.fft.irfft(spectrum, n=len(strain)).astype(float)


def _tukey_taper(n: int, alpha: float = 0.1) -> np.ndarray:
    """Tukey window.  Uses scipy if available, else manual."""
    if _SCIPY_AVAILABLE:
        return _tukey(n, alpha=alpha).astype(float)
    if n <= 1:
        return np.ones(n, dtype=float)
    w = np.ones(n, dtype=float)
    width = int(alpha * n / 2.0)
    if width > 0:
        t = np.linspace(0, np.pi, width, endpoint=False)
        taper = 0.5 * (1.0 - np.cos(t))
        w[:width] = taper
        w[-width:] = taper[::-1]
    return w


def _damped_sinusoid(
    t: np.ndarray, A: float, tau: float, f: float, phi: float, C: float,
) -> np.ndarray:
    """h(t) = A * exp(-t/tau) * cos(2*pi*f*t + phi) + C"""
    decay = np.exp(-t / tau)
    osc = np.cos(2.0 * np.pi * f * t + phi)
    return A * decay * osc + C


def _damped_sinusoid_residuals(
    params: np.ndarray, t: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    A, tau, f, phi, C = params
    return y - _damped_sinusoid(t, A, tau, f, phi, C)


def _damped_sinusoid_jac(
    params: np.ndarray, t: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Analytical Jacobian of *negative* residuals (dr/dp).
    Since r = y - model, dr/dp = -dmodel/dp."""
    A, tau, f, phi, C = params
    decay = np.exp(-t / tau)
    phase = 2.0 * np.pi * f * t + phi
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    n = t.size
    J = np.empty((n, 5), dtype=float)
    J[:, 0] = -(decay * cos_p)                             # dr/dA
    J[:, 1] = -(A * (t / (tau * tau)) * decay * cos_p)     # dr/dtau
    J[:, 2] = -(A * decay * (-2.0 * np.pi * t) * sin_p)   # dr/df
    J[:, 3] = -(A * decay * (-sin_p))                      # dr/dphi
    J[:, 4] = -np.ones(n, dtype=float)                     # dr/dC
    return J


def _estimate_initial_params(
    strain: np.ndarray, fs_hz: float, band_hz: list[float],
) -> tuple[float, float, float, float, float]:
    """Estimate initial [A, tau, f, phi, C] from strain.

    Key: frequency is estimated from a short window around the peak,
    not the full strain, so the Hanning window doesn't suppress the signal
    when it starts near t=0.
    """
    n = len(strain)
    # Find peak amplitude
    i_peak = int(np.argmax(np.abs(strain)))
    A_init = float(np.abs(strain[i_peak]))
    if A_init < 1e-30:
        A_init = 1e-10
    C_init = 0.0

    # f_init: FFT peak within band on a short window around the peak.
    # Use at least 256 samples or 10 cycles of the low band edge.
    lo_f, hi_f = band_hz[0], band_hz[1]
    min_win = max(256, int(10.0 / max(lo_f, 1.0) * fs_hz))
    fft_end = min(n, i_peak + min_win)
    fft_start = max(0, fft_end - min_win)
    fft_seg = strain[fft_start:fft_end]
    n_fft = len(fft_seg)
    if n_fft < 4:
        f_init = (lo_f + hi_f) / 2.0
    else:
        spectrum = np.fft.rfft(fft_seg * np.hanning(n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
        mask = (freqs >= lo_f) & (freqs <= hi_f)
        if not np.any(mask):
            f_init = (lo_f + hi_f) / 2.0
        else:
            band_mag = np.abs(spectrum[mask])
            f_init = float(freqs[mask][np.argmax(band_mag)])

    # tau_init: estimate from RMS envelope decay per half-cycle.
    tau_init = 0.01  # 10 ms default
    if f_init > 0:
        half_cycle = max(1, int(0.5 * fs_hz / f_init))
        rms_vals: list[float] = []
        rms_times: list[float] = []
        pos = i_peak
        while pos + half_cycle <= n:
            block = strain[pos : pos + half_cycle]
            rms = float(np.sqrt(np.mean(block ** 2)))
            rms_vals.append(rms)
            rms_times.append((pos - i_peak + half_cycle / 2.0) / fs_hz)
            pos += half_cycle
            if len(rms_vals) >= 20:
                break
        if len(rms_vals) >= 2 and rms_vals[0] > 0:
            half_rms = rms_vals[0] / 2.0
            for idx in range(1, len(rms_vals)):
                if rms_vals[idx] < half_rms:
                    dt = rms_times[idx]
                    if dt > 0:
                        tau_init = dt / math.log(2.0)
                    break
    tau_init = max(tau_init, 1e-5)
    phi_init = 0.0
    return A_init, tau_init, f_init, phi_init, C_init


def _qnm_fit_detector(
    strain: np.ndarray,
    fs_hz: float,
    band_hz: list[float],
    whitened: bool = False,
) -> dict[str, Any]:
    """Run QNM damped-sinusoid fit on one detector's strain.

    The fit is restricted to the signal region: from the peak amplitude to
    a window spanning several expected decay constants, avoiding noise-dominated
    samples that would bias the optimizer.

    Returns dict with keys:
      f_qnm_hz, tau_qnm_s, Q_qnm, sigma_f, sigma_tau,
      rmse, chi2_red, n_samples, status, notes
    """
    notes: list[str] = []
    n_total = int(strain.size)
    result: dict[str, Any] = {
        "f_qnm_hz": None,
        "f_bounds_hz": [float(band_hz[0]), float(band_hz[1])],
        "clipped_f": False,
        "tau_qnm_s": None,
        "tau_bounds_s": [TAU_MIN_S, TAU_MAX_S],
        "clipped_tau": False,
        "Q_qnm": None,
        "sigma_f": None,
        "sigma_tau": None,
        "rmse": None,
        "chi2_red": None,
        "n_samples": n_total,
        "status": "FAIL",
        "notes": notes,
    }

    if not _SCIPY_AVAILABLE:
        notes.append("scipy not available; QNM fit skipped")
        return result

    if n_total < 32:
        notes.append(f"strain too short for QNM fit (n={n_total})")
        return result

    lo, hi = float(band_hz[0]), float(band_hz[1])

    # --- Initial parameter estimation on raw signal ---
    # No taper/filter here: the ringdown window already starts at t0 and
    # the signal peak may be at the very beginning.
    try:
        A0, tau0, f0, phi0, C0 = _estimate_initial_params(strain, fs_hz, [lo, hi])
    except Exception as exc:
        notes.append(f"initial param estimation failed: {exc}")
        return result

    # --- Truncate to signal region ---
    # Fit from peak to ~8*tau_init to avoid fitting noise-dominated tail.
    i_peak = int(np.argmax(np.abs(strain)))
    fit_len = max(64, int(8.0 * tau0 * fs_hz))
    fit_end = min(n_total, i_peak + fit_len)
    if fit_end - i_peak < 32:
        fit_end = min(n_total, i_peak + 32)

    y_fit = strain[i_peak:fit_end].copy().astype(float)
    n_fit = int(y_fit.size)
    t_fit = np.arange(n_fit, dtype=float) / fs_hz

    # No taper: the damped sinusoid naturally decays to zero,
    # so no windowing is needed and tapering would distort the envelope.

    # Re-estimate A from the fit window
    A0 = float(np.max(np.abs(y_fit - np.mean(y_fit))))
    if A0 < 1e-30:
        A0 = 1e-10
    C0 = float(np.mean(y_fit))
    p0 = np.array([A0, tau0, f0, phi0, C0])

    # Bounds: A free, tau in contract range, f in analysis band, phi in [-2pi, 2pi], C free
    lower = [-np.inf, TAU_MIN_S, lo, -2.0 * np.pi, -np.inf]
    upper = [np.inf, TAU_MAX_S, hi, 2.0 * np.pi, np.inf]

    # Clamp initial guess inside bounds
    for i in range(len(p0)):
        p0[i] = max(lower[i], min(upper[i], p0[i]))

    # --- Fit ---
    try:
        res = _least_squares(
            _damped_sinusoid_residuals,
            p0,
            jac=_damped_sinusoid_jac,
            args=(t_fit, y_fit),
            bounds=(lower, upper),
            method="trf",
            max_nfev=_QNM_MAX_NFEV,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
    except Exception as exc:
        notes.append(f"least_squares failed: {exc}")
        return result

    if not res.success and res.status not in (1, 2, 3, 4):
        notes.append(f"least_squares did not converge: {res.message}")
        return result

    A_fit, tau_fit, f_fit, phi_fit, C_fit = res.x

    if not math.isfinite(tau_fit) or tau_fit <= 0:
        notes.append(f"fitted tau non-positive or non-finite: {tau_fit}")
        return result
    if not math.isfinite(f_fit):
        notes.append(f"fitted f non-finite: {f_fit}")
        return result

    # --- Goodness of fit ---
    residuals = res.fun
    ss_res = float(np.sum(residuals ** 2))
    n_params = 5
    dof = max(n_fit - n_params, 1)
    rmse = float(np.sqrt(ss_res / n_fit))
    chi2_red = float(ss_res / dof)
    Q_qnm = float(math.pi * f_fit * tau_fit)

    result["f_qnm_hz"] = float(f_fit)
    result["tau_qnm_s"] = float(tau_fit)
    result["clipped_f"] = _is_clipped(float(f_fit), lo, hi, CLIP_EPS_FRAC)
    result["clipped_tau"] = _is_clipped(float(tau_fit), TAU_MIN_S, TAU_MAX_S, CLIP_EPS_FRAC)
    result["Q_qnm"] = float(Q_qnm)
    result["rmse"] = rmse
    result["chi2_red"] = chi2_red
    result["n_samples"] = n_fit

    # --- Uncertainty estimation: C ≈ sigma^2 * (J^T J)^-1 ---
    sigma_f: float | None = None
    sigma_tau: float | None = None
    try:
        J = res.jac
        JtJ = J.T @ J
        cov = np.linalg.inv(JtJ) * (ss_res / dof)
        variances = np.diag(cov)
        if variances[1] >= 0:
            sigma_tau = float(np.sqrt(variances[1]))
        if variances[2] >= 0:
            sigma_f = float(np.sqrt(variances[2]))
        if sigma_tau is not None and not math.isfinite(sigma_tau):
            sigma_tau = None
            notes.append("sigma_tau non-finite, set to null")
        if sigma_f is not None and not math.isfinite(sigma_f):
            sigma_f = None
            notes.append("sigma_f non-finite, set to null")
    except Exception as exc:
        notes.append(f"covariance estimation failed: {exc}")

    result["sigma_f"] = sigma_f
    result["sigma_tau"] = sigma_tau
    result["status"] = "OK"

    return result


def _build_decision_qnm(
    qnm_fit: dict[str, Any], band_hz: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build decision_qnm verdict from per-detector QNM fit results."""

    def _set_inspect() -> None:
        nonlocal verdict
        if verdict == "PASS":
            verdict = "INSPECT"

    detector_ids = sorted(
        det for det, det_fit in qnm_fit.items() if isinstance(det_fit, dict)
    )

    def _valid_tau_sigma_entries(
        entries: dict[str, Any],
    ) -> list[tuple[str, float, float]]:
        """Return finite (detector, tau, sigma_tau) tuples from qnm_fit map."""
        valid: list[tuple[str, float, float]] = []
        for det, det_fit in entries.items():
            if not isinstance(det_fit, dict):
                continue
            if det_fit.get("status") != "OK":
                continue
            tau_raw = det_fit.get("tau_qnm_s")
            sigma_raw = det_fit.get("sigma_tau")
            if tau_raw is None or sigma_raw is None:
                continue
            try:
                tau_val = float(tau_raw)
                sigma_val = float(sigma_raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(tau_val) and math.isfinite(sigma_val) and sigma_val > 0:
                valid.append((det, tau_val, sigma_val))
        return valid

    def _compute_pairwise_tau_zscores(
        entries: dict[str, Any],
    ) -> list[dict[str, float | str]]:
        """Compute pairwise detector tau consistency z-scores using combined sigma."""
        valid = _valid_tau_sigma_entries(entries)
        rows: list[dict[str, float | str]] = []
        for idx_a in range(len(valid)):
            det_a, tau_a, sigma_a = valid[idx_a]
            for idx_b in range(idx_a + 1, len(valid)):
                det_b, tau_b, sigma_b = valid[idx_b]
                sigma_comb = math.sqrt(sigma_a * sigma_a + sigma_b * sigma_b)
                if not math.isfinite(sigma_comb) or sigma_comb <= 0:
                    continue
                z = abs(tau_a - tau_b) / sigma_comb
                rows.append(
                    {
                        "detector_a": det_a,
                        "detector_b": det_b,
                        "z_tau": float(z),
                        "sigma_comb": float(sigma_comb),
                    }
                )
        return rows

    def _compute_consensus_leave_one_out(
        entries: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute weighted consensus tau and leave-one-out influence diagnostics."""
        valid = _valid_tau_sigma_entries(entries)
        if len(valid) < 2:
            return {
                "consensus_tau_s": None,
                "consensus_sigma_s": None,
                "leave_one_out": [],
                "max_influence_z": None,
            }

        weights = [1.0 / (sigma * sigma) for _, _, sigma in valid]
        wsum = float(sum(weights))
        if wsum <= 0 or not math.isfinite(wsum):
            return {
                "consensus_tau_s": None,
                "consensus_sigma_s": None,
                "leave_one_out": [],
                "max_influence_z": None,
            }

        consensus_tau = float(
            sum(w * tau for w, (_, tau, _) in zip(weights, valid)) / wsum
        )
        consensus_sigma = float(math.sqrt(1.0 / wsum))

        leave_one_out: list[dict[str, float | str]] = []
        max_influence_z: float | None = None
        for det_excluded, _, _ in valid:
            loo_valid = [row for row in valid if row[0] != det_excluded]
            if len(loo_valid) < 1:
                continue
            loo_weights = [1.0 / (sigma * sigma) for _, _, sigma in loo_valid]
            loo_wsum = float(sum(loo_weights))
            if loo_wsum <= 0 or not math.isfinite(loo_wsum):
                continue
            loo_tau = float(
                sum(w * tau for w, (_, tau, _) in zip(loo_weights, loo_valid)) / loo_wsum
            )
            loo_sigma = float(math.sqrt(1.0 / loo_wsum))
            denom = math.sqrt(consensus_sigma * consensus_sigma + loo_sigma * loo_sigma)
            influence_z = None
            if denom > 0 and math.isfinite(denom):
                influence_z = float(abs(loo_tau - consensus_tau) / denom)
                if max_influence_z is None or influence_z > max_influence_z:
                    max_influence_z = influence_z
            leave_one_out.append(
                {
                    "excluded_detector": det_excluded,
                    "tau_loo_s": loo_tau,
                    "sigma_loo_s": loo_sigma,
                    "influence_z": influence_z,
                }
            )

        return {
            "consensus_tau_s": consensus_tau,
            "consensus_sigma_s": consensus_sigma,
            "leave_one_out": leave_one_out,
            "max_influence_z": max_influence_z,
        }

    def _compute_window_stability(entries: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate per-detector stability from tau/sigma window replicas if present."""
        out: list[dict[str, Any]] = []
        for det, det_fit in entries.items():
            if not isinstance(det_fit, dict):
                continue
            replicas_raw = det_fit.get("window_replicas")
            if not isinstance(replicas_raw, list) or len(replicas_raw) < 2:
                continue
            tau_vals: list[float] = []
            sigma_vals: list[float] = []
            for replica in replicas_raw:
                if not isinstance(replica, dict):
                    continue
                tau_raw = replica.get("tau_qnm_s")
                sigma_raw = replica.get("sigma_tau")
                if tau_raw is None or sigma_raw is None:
                    continue
                try:
                    tau_val = float(tau_raw)
                    sigma_val = float(sigma_raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(tau_val):
                    tau_vals.append(tau_val)
                if math.isfinite(sigma_val) and sigma_val > 0:
                    sigma_vals.append(sigma_val)
            if len(tau_vals) < 2 or not sigma_vals:
                continue
            std_tau = float(np.std(np.asarray(tau_vals, dtype=float), ddof=0))
            med_sigma = float(np.median(np.asarray(sigma_vals, dtype=float)))
            ratio = float(std_tau / med_sigma) if med_sigma > 0 else None
            out.append(
                {
                    "detector": det,
                    "std_tau_replicas_s": std_tau,
                    "median_sigma_tau_replicas_s": med_sigma,
                    "stability_ratio": ratio,
                    "stable": (ratio is not None and ratio <= DEFAULT_STABILITY_K),
                    "stability_k": DEFAULT_STABILITY_K,
                }
            )
        return out

    reasons: list[str] = []
    verdict = "PASS"
    tau_h1: float | None = None
    tau_l1: float | None = None
    status_h1 = None
    status_l1 = None

    for det in detector_ids:
        det_fit = qnm_fit.get(det)
        if det_fit is None:
            verdict = "FAIL"
            reasons.append(f"{det}: qnm_fit missing")
            continue
        if det_fit.get("status") != "OK":
            if verdict == "PASS":
                verdict = "INSPECT"
            reasons.append(f"{det}: qnm_fit status={det_fit.get('status')}")
            continue
        f = det_fit.get("f_qnm_hz")
        tau = det_fit.get("tau_qnm_s")
        if det == "H1":
            status_h1 = det_fit.get("status")
            tau_h1 = tau
        elif det == "L1":
            status_l1 = det_fit.get("status")
            tau_l1 = tau
        if f is None or tau is None:
            _set_inspect()
            reasons.append(f"{det}: f_qnm_hz or tau_qnm_s is null despite status OK")
            continue
        if not math.isfinite(float(f)) or not math.isfinite(float(tau)):
            _set_inspect()
            reasons.append(f"{det}: nan/non-finite qnm fit value")
            continue
        if tau <= 0:
            _set_inspect()
            reasons.append(f"{det}: tau_qnm_s <= 0 ({tau})")
        if f < band_hz[0] or f > band_hz[1]:
            _set_inspect()
            reasons.append(f"{det}: f_qnm_hz={f} outside band {band_hz}")
        # Check for huge uncertainties
        sigma_f = det_fit.get("sigma_f")
        sigma_tau = det_fit.get("sigma_tau")
        if sigma_f is not None and not math.isfinite(float(sigma_f)):
            _set_inspect()
            reasons.append(f"{det}: sigma_f is nan/non-finite")
        if sigma_tau is not None and not math.isfinite(float(sigma_tau)):
            _set_inspect()
            reasons.append(f"{det}: sigma_tau is nan/non-finite")
        if sigma_f is not None and f > 0 and sigma_f / f > 0.5:
            _set_inspect()
            reasons.append(f"{det}: sigma_f/f > 50% ({sigma_f/f:.2f})")
        if sigma_tau is not None and tau > 0 and sigma_tau / tau > 0.5:
            _set_inspect()
            reasons.append(f"{det}: sigma_tau/tau > 50% ({sigma_tau/tau:.2f})")

        clipping_fraction = det_fit.get("clipping_fraction")
        if clipping_fraction is not None:
            try:
                cf = float(clipping_fraction)
            except (TypeError, ValueError):
                cf = float("nan")
            if not math.isfinite(cf):
                _set_inspect()
                reasons.append(f"{det}: clipping_fraction is nan/non-finite")
            elif cf > DEFAULT_CLIPPING_FRACTION_MAX:
                _set_inspect()
                reasons.append(
                    f"{det}: clipping_fraction={cf:.3f} exceeds {DEFAULT_CLIPPING_FRACTION_MAX:.3f}"
                )

        if det_fit.get("clipped_tau") is True:
            reasons.append(
                f"{det}: tau_qnm_s clipped to bounds {det_fit.get('tau_bounds_s')}"
            )

    tau_mean: float | None = None
    tau_frac_diff: float | None = None
    if (
        status_h1 == "OK"
        and status_l1 == "OK"
        and tau_h1 is not None
        and tau_l1 is not None
    ):
        tau_mean = 0.5 * (float(tau_h1) + float(tau_l1))
        if tau_mean > 0:
            tau_frac_diff = abs(float(tau_h1) - float(tau_l1)) / tau_mean
            if tau_frac_diff > TAU_FRAC_DIFF_MAX:
                _set_inspect()
                reasons.append(
                    f"tau inconsistente H1/L1: frac_diff={tau_frac_diff:.3f} (>0.2)"
                )

    pairwise_tau_zscores = _compute_pairwise_tau_zscores(qnm_fit)
    if any(float(row["z_tau"]) > DEFAULT_LOO_Z_THRESHOLD for row in pairwise_tau_zscores):
        _set_inspect()
        reasons.append(
            f"pairwise tau consistency z-score exceeds {DEFAULT_LOO_Z_THRESHOLD:.1f}"
        )

    leave_one_out = _compute_consensus_leave_one_out(qnm_fit)
    max_influence_z = leave_one_out.get("max_influence_z")
    if isinstance(max_influence_z, float) and max_influence_z > DEFAULT_LOO_Z_THRESHOLD:
        _set_inspect()
        reasons.append(
            f"leave-one-out detector influence exceeds {DEFAULT_LOO_Z_THRESHOLD:.1f}"
        )

    stability = _compute_window_stability(qnm_fit)
    if any(row.get("stable") is False for row in stability):
        _set_inspect()
        reasons.append("window-replica stability check failed")

    qnm_consistency = {
        "tau_mean_s": tau_mean,
        "tau_frac_diff": tau_frac_diff,
        "tau_frac_diff_max": TAU_FRAC_DIFF_MAX,
        "pairwise_tau_zscores": pairwise_tau_zscores,
        "pairwise_z_threshold": DEFAULT_LOO_Z_THRESHOLD,
        "leave_one_out": leave_one_out,
        "window_stability": stability,
    }
    return {"verdict": verdict, "reasons": reasons}, qnm_consistency


def _write_failure(
    stage_dir: Path,
    stage_name: str,
    run_id: str,
    params: dict[str, Any],
    inputs: list[dict[str, str]],
    reason: str,
) -> None:
    summary = {
        "stage": stage_name,
        "run": run_id,
        "created": utc_now_iso(),
        "version": "v0",
        "parameters": params,
        "inputs": inputs,
        "outputs": [],
        "verdict": "FAIL",
        "error": reason,
    }
    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {"stage_summary": summary_path},
        extra={"inputs": inputs, "verdict": "FAIL", "error": reason},
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Canonical stage: minimal physical inference on real ringdown window"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument(
        "--stage-name",
        default=STAGE_NAME_DEFAULT,
        help=f"stage name (default: {STAGE_NAME_DEFAULT})",
    )
    ap.add_argument(
        "--window-stage",
        default="ringdown_real_ringdown_window",
        help="stage name for ringdown window inputs",
    )
    ap.add_argument("--band-hz", default="150,400", type=_parse_band_hz)
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        _abort(str(exc))

    stage_name = args.stage_name
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, stage_name, base_dir=out_root)

    inputs_dir = run_dir / args.window_stage / "outputs"
    input_paths = {
        "features": run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        "observables": run_dir
        / "ringdown_real_observables_v0"
        / "outputs"
        / "observables.jsonl",
        "H1": inputs_dir / "H1_rd.npz",
        "L1": inputs_dir / "L1_rd.npz",
        "segments": inputs_dir / "segments_rd.json",
    }

    params = {
        "run": args.run,
        "stage_name": stage_name,
        "window_stage": args.window_stage,
        "band_hz": args.band_hz,
    }
    inputs_list: list[dict[str, str]] = []
    missing = []
    for path in input_paths.values():
        if not path.exists():
            missing.append(str(path))
            inputs_list.append({"path": str(path.relative_to(run_dir)), "sha256": ""})
        else:
            inputs_list.append(
                {"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path)}
            )

    if missing:
        reason = f"missing inputs: {', '.join(missing)}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    try:
        features = _read_single_jsonl(input_paths["features"])
        observables = _read_single_jsonl(input_paths["observables"])
    except Exception as exc:
        reason = f"no se pudo leer features/observables: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    detectors = observables.get("detectors")
    if not isinstance(detectors, list):
        reason = "observables missing detectors list"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)
    if "H1" not in detectors or "L1" not in detectors:
        reason = "observables must include detectors H1 and L1"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    fs_hz_fallback = observables.get("fs_hz")
    if fs_hz_fallback is None:
        fs_hz_fallback = features.get("fs_hz") if isinstance(features, dict) else None
    if fs_hz_fallback is None:
        reason = "fs_hz missing in observables/features"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)
    try:
        fs_hz_fallback = _require_positive(float(fs_hz_fallback), "fs_hz")
    except Exception as exc:
        reason = f"fs_hz inválido: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    n_samples = observables.get("n_samples")
    if not isinstance(n_samples, dict) or "H1" not in n_samples or "L1" not in n_samples:
        reason = "observables.n_samples missing H1/L1"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    t0_gps = observables.get("t0_gps")
    if t0_gps is not None:
        try:
            t0_gps = _require_finite(float(t0_gps), "t0_gps")
        except Exception as exc:
            reason = f"t0_gps inválido: {exc}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

    fit: dict[str, dict[str, Any]] = {}
    window: dict[str, Any] = {
        "stage": args.window_stage,
        "fs_hz": None,
        "duration_s": {},
        "n_samples": {},
    }
    decision_reasons: list[str] = []
    decision_verdict = "PASS"
    contract_verdict = "PASS"
    contract_reasons: list[str] = []
    tau_estimator_det: dict[str, dict[str, Any]] = {}
    strains: dict[str, tuple[np.ndarray, float]] = {}

    for det in ["H1", "L1"]:
        try:
            strain, fs_det, fs_source = _load_strain_and_fs(
                input_paths[det], fs_hz_fallback
            )
            f_peak_hz, peak_mag, df_hz = _estimate_f_peak(strain, fs_det, args.band_hz)
        except Exception as exc:
            reason = f"no se pudo estimar f_peak para {det}: {exc}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

        strains[det] = (strain, fs_det)

        notes: list[str] = []
        if window["fs_hz"] is None:
            window["fs_hz"] = fs_det
        elif abs(fs_det - float(window["fs_hz"])) > 1e-12:
            contract_verdict = "INSPECT"
            contract_reasons.append(
                f"{det}: fs_hz mismatch (npz={fs_det}, window={window['fs_hz']})"
            )
            notes.append(f"WARN: fs_hz distinto ({fs_det} vs {window['fs_hz']})")
        if fs_source != "npz":
            contract_verdict = "INSPECT"
            contract_reasons.append(f"{det}: fs_hz inferred from observables/features")

        tau_s, tau_notes, tau_metrics = _estimate_tau(strain, fs_det)
        notes.extend(tau_notes)
        tau_estimator_det[det] = tau_metrics

        q_val = None
        if tau_s is not None:
            q_val = float(math.pi * f_peak_hz * tau_s)
            if not math.isfinite(q_val):
                notes.append("WARN: Q no finito")
                q_val = None

        if tau_s is None:
            decision_verdict = "INSPECT"
            nv = tau_metrics["n_blocks_valid"]
            if nv < MIN_BLOCKS_TAU:
                decision_reasons.append(
                    f"{det}: tau_s no estimado (n_blocks_valid={nv} < {MIN_BLOCKS_TAU})"
                )
            else:
                decision_reasons.append(f"{det}: tau_s no estimado")

        n_samples_det = int(strain.size)
        window["n_samples"][det] = n_samples_det
        fs_ref = float(window["fs_hz"])
        window["duration_s"][det] = n_samples_det / fs_ref

        fit[det] = {
            "n_samples": int(strain.size),
            "df_hz": df_hz,
            "f_peak_hz": f_peak_hz,
            "peak_mag": peak_mag,
            "tau_s": tau_s,
            "Q": q_val,
            "notes": notes,
        }

        expected_df = fs_ref / n_samples_det
        if abs(df_hz - expected_df) > 1e-12:
            contract_verdict = "INSPECT"
            contract_reasons.append(
                f"{det}: df_hz inconsistente (df_hz={df_hz}, esperado={expected_df})"
            )

        expected_duration = n_samples_det / fs_ref
        if abs(window["duration_s"][det] - expected_duration) > 1e-12:
            contract_verdict = "INSPECT"
            contract_reasons.append(
                f"{det}: duration_s inconsistente (duration_s={window['duration_s'][det]}, esperado={expected_duration})"
            )

    if window["fs_hz"] is None:
        contract_verdict = "INSPECT"
        contract_reasons.append("fs_hz missing from window npz inputs")
        window["fs_hz"] = float(fs_hz_fallback)

    features_payload = {
        key: features.get(key)
        for key in ["snr_proxy", "rms", "peak_abs"]
        if isinstance(features, dict) and key in features
    }
    if isinstance(features, dict) and "duration_s" in features:
        features_payload["features_duration_s"] = features.get("duration_s")

    # --- tau_estimator audit block ---
    fs_report = float(window["fs_hz"])
    block_samples = max(16, int(0.01 * fs_report))
    tau_estimator: dict[str, Any] = {
        "block_len_s": block_samples / fs_report,
        "min_blocks_required": MIN_BLOCKS_TAU,
    }
    for det_key, det_metrics in tau_estimator_det.items():
        tau_estimator[det_key] = {
            "n_blocks_total": det_metrics["n_blocks_total"],
            "n_blocks_valid": det_metrics["n_blocks_valid"],
            "reject_reasons": det_metrics["reject_reasons"],
        }

    # Contract: if fit.<IFO>.tau_s is null, tau_estimator.<IFO> must exist
    for det in ["H1", "L1"]:
        if fit[det]["tau_s"] is None:
            te_det = tau_estimator.get(det)
            if (
                te_det is None
                or "n_blocks_valid" not in te_det
            ):
                contract_verdict = "INSPECT"
                contract_reasons.append(
                    f"{det}: tau_estimator metrics missing for null tau_s"
                )

    # --- QNM damped-sinusoid fit ---
    band_hz_list = [float(args.band_hz[0]), float(args.band_hz[1])]
    qnm_fit: dict[str, Any] = {
        "model": QNM_MODEL_VERSION,
        "band_hz": band_hz_list,
    }
    for det in ["H1", "L1"]:
        det_strain, det_fs = strains[det]
        qnm_result = _qnm_fit_detector(det_strain, det_fs, band_hz_list)
        qnm_fit[det] = qnm_result
        # Add to decision.reasons if qnm_fit fails but the rest passes
        if qnm_result["status"] != "OK":
            qnm_notes = "; ".join(qnm_result["notes"]) if qnm_result["notes"] else "unknown"
            decision_reasons.append(
                f"{det}: qnm_fit FAIL ({qnm_notes})"
            )

    # --- QNM contract checks ---
    for det in ["H1", "L1"]:
        det_qnm = qnm_fit[det]
        if det_qnm["status"] == "OK":
            # Contract: if status OK => f_qnm_hz and tau_qnm_s not null and tau>0
            if det_qnm["f_qnm_hz"] is None or det_qnm["tau_qnm_s"] is None:
                contract_verdict = "INSPECT"
                contract_reasons.append(
                    f"{det}: qnm_fit status OK but f_qnm_hz or tau_qnm_s is null"
                )
            if det_qnm["tau_bounds_s"] is None or det_qnm["clipped_tau"] is None:
                contract_verdict = "INSPECT"
                contract_reasons.append(
                    f"{det}: qnm_fit status OK but tau_bounds_s or clipped_tau is null"
                )
            elif det_qnm["tau_qnm_s"] <= 0:
                contract_verdict = "INSPECT"
                contract_reasons.append(
                    f"{det}: qnm_fit status OK but tau_qnm_s <= 0"
                )
        elif det_qnm["status"] == "FAIL":
            # Contract: if status FAIL => notes not empty
            if not det_qnm["notes"]:
                contract_verdict = "INSPECT"
                contract_reasons.append(
                    f"{det}: qnm_fit status FAIL but notes empty"
                )

    decision_qnm, qnm_consistency = _build_decision_qnm(qnm_fit, band_hz_list)

    report = {
        "run_id": args.run,
        "t0_gps": t0_gps,
        "fs_hz": fs_report,
        "band_hz": band_hz_list,
        "features": features_payload,
        "window": window,
        "fit": fit,
        "tau_estimator": tau_estimator,
        "qnm_fit": qnm_fit,
        "qnm_consistency": qnm_consistency,
        "decision": {
            "verdict": decision_verdict,
            "reasons": decision_reasons,
        },
        "decision_qnm": decision_qnm,
    }

    report_path = outputs_dir / "inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    verdict_path = outputs_dir / "contract_verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(
            {"verdict": contract_verdict, "reasons": contract_reasons},
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    outputs_list = [
        {"path": str(report_path.relative_to(run_dir)), "sha256": sha256_file(report_path)},
        {"path": str(verdict_path.relative_to(run_dir)), "sha256": sha256_file(verdict_path)},
    ]

    summary = {
        "stage": stage_name,
        "run": args.run,
        "created": utc_now_iso(),
        "version": "v0",
        "parameters": params,
        "inputs": inputs_list,
        "outputs": outputs_list,
        "verdict": "PASS",
        "format": {"inference_report": "single_record_per_run", "detectors": ["H1", "L1"]},
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "inference_report": report_path,
            "contract_verdict": verdict_path,
            "stage_summary": summary_path,
        },
        extra={"inputs": inputs_list},
    )

    print(f"OK: {stage_name} PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
