#!/usr/bin/env python3
"""MVP Stage 3: Estimate ringdown observables (f, tau, Q) per event.

CLI:
    python mvp/s3_ringdown_estimates.py --run <run_id> \
        [--band-low 150] [--band-high 400]

Inputs (from s2_ringdown_window):
    runs/<run>/s2_ringdown_window/outputs/{H1,L1}_rd.npz

Outputs (runs/<run>/s3_ringdown_estimates/outputs/):
    estimates.json       Per-detector and combined estimates of f, tau, Q

Contracts:
    - At least one detector must produce a valid estimate.
    - f_hz must be within [band_low, band_high].
    - tau_s must be > 0 and finite.
    - Q = pi * f * tau must be > 0.

Method: Hilbert-envelope analysis.
    1. Bandpass filter in [band_low, band_high] Hz.
    2. Analytic signal via Hilbert transform.
    3. Instantaneous frequency from phase derivative (median in band).
    4. Decay time from log-envelope linear fit after peak.
    5. Q = pi * f * tau.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "s3_ringdown_estimates"
UPSTREAM_STAGE = "s2_ringdown_window"
EXIT_CONTRACT_FAIL = 2


def _abort(message: str) -> None:
    print(f"[{STAGE_NAME}] ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def estimate_ringdown_observables(
    strain: np.ndarray,
    fs: float,
    band_low: float = 150.0,
    band_high: float = 400.0,
) -> dict[str, float]:
    """Estimate f, tau, Q from a ringdown strain segment.

    Returns dict with keys: f_hz, tau_s, Q, snr_peak.
    Raises ValueError if estimation fails.
    """
    from scipy.signal import butter, sosfilt, hilbert

    n = strain.size
    if n < 16:
        raise ValueError(f"Strain too short for analysis: {n} samples")

    nyquist = fs / 2.0
    if band_high >= nyquist:
        band_high = nyquist * 0.95
    if band_low >= band_high:
        raise ValueError(f"Invalid band: [{band_low}, {band_high}] Hz (Nyquist={nyquist})")

    # 1. Bandpass filter
    sos = butter(4, [band_low, band_high], btype="band", fs=fs, output="sos")
    filtered = sosfilt(sos, strain)

    # 2. Analytic signal
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))

    # 3. Instantaneous frequency (median of valid values)
    inst_freq = np.diff(inst_phase) * fs / (2.0 * np.pi)
    valid_mask = (inst_freq > band_low * 0.8) & (inst_freq < band_high * 1.2)
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient valid frequency samples")
    f_hz = float(np.median(inst_freq[valid_mask]))

    # 4. Decay time from log-envelope fit
    peak_idx = int(np.argmax(envelope))
    snr_peak = float(envelope[peak_idx] / (np.std(envelope[:max(1, peak_idx // 2)]) + 1e-30))

    # Fit region: from peak to end (or until envelope drops below noise floor)
    noise_floor = np.median(envelope) * 0.1 + 1e-30
    fit_mask = np.arange(n) >= peak_idx
    fit_mask &= envelope > noise_floor

    fit_indices = np.flatnonzero(fit_mask)
    if fit_indices.size < 5:
        raise ValueError(f"Insufficient samples for tau fit: {fit_indices.size}")

    t_fit = fit_indices.astype(float) / fs
    log_env = np.log(envelope[fit_indices])

    # Robust linear fit: log(A*exp(-t/tau)) = log(A) - t/tau
    coeffs = np.polyfit(t_fit - t_fit[0], log_env, 1)
    gamma = -coeffs[0]  # decay rate (1/tau)

    if gamma <= 0:
        raise ValueError(f"Non-decaying signal: gamma={gamma:.4f}")

    tau_s = 1.0 / gamma
    Q = math.pi * f_hz * tau_s

    if Q <= 0 or not math.isfinite(Q):
        raise ValueError(f"Invalid Q={Q:.4f}")

    return {"f_hz": f_hz, "tau_s": tau_s, "Q": Q, "snr_peak": snr_peak}


def _write_failure(stage_dir: Path, run_id: str, params: dict, inputs: list, reason: str) -> None:
    summary = {
        "stage": STAGE_NAME, "run": run_id, "created": utc_now_iso(),
        "version": "v1", "parameters": params, "inputs": inputs, "outputs": [],
        "verdict": "FAIL", "error": reason,
    }
    sp = write_stage_summary(stage_dir, summary)
    write_manifest(stage_dir, {"stage_summary": sp}, extra={"verdict": "FAIL", "error": reason})


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE_NAME}: estimate f, tau, Q")
    ap.add_argument("--run", required=True)
    ap.add_argument("--band-low", type=float, default=150.0, help="Lower bandpass frequency (Hz)")
    ap.add_argument("--band-high", type=float, default=400.0, help="Upper bandpass frequency (Hz)")
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = out_root / args.run

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        _abort(str(exc))

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)

    # Discover detector files from s2
    upstream_dir = run_dir / UPSTREAM_STAGE / "outputs"
    detector_files: dict[str, Path] = {}
    for candidate in ("H1_rd.npz", "L1_rd.npz", "V1_rd.npz"):
        p = upstream_dir / candidate
        if p.exists():
            det = candidate.split("_")[0]
            detector_files[det] = p

    if not detector_files:
        _abort(f"No detector files found in {upstream_dir}")

    # Load window metadata
    window_meta_path = upstream_dir / "window_meta.json"
    window_meta: dict[str, Any] = {}
    if window_meta_path.exists():
        with open(window_meta_path, "r", encoding="utf-8") as f:
            window_meta = json.load(f)

    params: dict[str, Any] = {
        "band_low_hz": args.band_low,
        "band_high_hz": args.band_high,
        "method": "hilbert_envelope",
    }
    inputs_list = [
        {"path": str(p.relative_to(run_dir)), "sha256": sha256_file(p)}
        for p in detector_files.values()
    ]

    try:
        per_detector: dict[str, dict[str, float]] = {}
        valid_estimates: list[dict[str, float]] = []

        for det, det_path in detector_files.items():
            data = np.load(det_path)
            strain = np.asarray(data["strain"], dtype=np.float64)
            fs = float(np.asarray(data["sample_rate_hz"]).flat[0])

            if strain.ndim != 1:
                _abort(f"{det}: strain is not 1-D")
            if not np.all(np.isfinite(strain)):
                _abort(f"{det}: strain contains NaN/Inf")

            try:
                est = estimate_ringdown_observables(strain, fs, args.band_low, args.band_high)
                per_detector[det] = est
                valid_estimates.append(est)
            except ValueError as exc:
                per_detector[det] = {"error": str(exc)}

        if not valid_estimates:
            _abort("No detector produced a valid estimate")

        # Combined estimate: weighted average by SNR (or simple average)
        weights = np.array([e.get("snr_peak", 1.0) for e in valid_estimates])
        weights = weights / weights.sum()
        combined_f = float(sum(w * e["f_hz"] for w, e in zip(weights, valid_estimates)))
        combined_tau = float(sum(w * e["tau_s"] for w, e in zip(weights, valid_estimates)))
        combined_Q = math.pi * combined_f * combined_tau

        estimates = {
            "schema_version": "mvp_estimates_v1",
            "event_id": window_meta.get("event_id", "unknown"),
            "method": "hilbert_envelope",
            "band_hz": [args.band_low, args.band_high],
            "combined": {
                "f_hz": combined_f,
                "tau_s": combined_tau,
                "Q": combined_Q,
            },
            "per_detector": per_detector,
            "n_detectors_valid": len(valid_estimates),
        }

        est_path = outputs_dir / "estimates.json"
        write_json_atomic(est_path, estimates)

        outputs_list = [{"path": str(est_path.relative_to(run_dir)), "sha256": sha256_file(est_path)}]
        summary = {
            "stage": STAGE_NAME, "run": args.run, "created": utc_now_iso(),
            "version": "v1", "parameters": params, "inputs": inputs_list,
            "outputs": outputs_list, "verdict": "PASS",
            "results": estimates["combined"],
        }
        sp = write_stage_summary(stage_dir, summary)
        write_manifest(
            stage_dir,
            {"estimates": est_path, "stage_summary": sp},
            extra={"inputs": inputs_list},
        )

        print(f"OK: {STAGE_NAME} PASS (f={combined_f:.1f} Hz, Q={combined_Q:.1f})")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        _write_failure(stage_dir, args.run, params, inputs_list, str(exc))
        _abort(str(exc))
        return EXIT_CONTRACT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
