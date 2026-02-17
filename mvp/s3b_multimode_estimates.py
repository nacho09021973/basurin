#!/usr/bin/env python3
"""Stage s3b_multimode_estimates: canonical multi-mode estimates (220,221)."""
from __future__ import annotations

import argparse
import json
import math
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
from mvp.s3_ringdown_estimates import estimate_ringdown_observables

STAGE = "s3b_multimode_estimates"
TARGET_MODES = [
    {"mode": [2, 2, 0], "label": "220"},
    {"mode": [2, 2, 1], "label": "221"},
]


def compute_covariance(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must be shape (n,2)")
    if samples.shape[0] < 2:
        raise ValueError("need >=2 samples for covariance")
    return np.cov(samples, rowvar=False, ddof=1)


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


def _bootstrap_mode_log_samples(
    signal: np.ndarray,
    fs: float,
    estimator: Callable[[np.ndarray, float], dict[str, float]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    base = estimator(signal, fs)
    block = max(16, int(fs / max(base["f_hz"], 1.0)))
    n = signal.size
    n_blocks = max(1, n // block)
    draws = -(-n // block)

    samples: list[list[float]] = []
    failed = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_blocks, size=draws)
        resampled = np.concatenate([signal[i * block:(i + 1) * block] for i in idx])[:n]
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
    return np.asarray(samples, dtype=float), failed


def _estimate_220(signal: np.ndarray, fs: float) -> dict[str, float]:
    return estimate_ringdown_observables(signal, fs)


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


def _estimate_221_from_signal(signal: np.ndarray, fs: float) -> dict[str, float]:
    est220 = _estimate_220(signal, fs)
    residual = signal - _template_220(signal, fs, est220)
    return estimate_ringdown_observables(residual, fs)


def _mode_payload(label: str, mode: list[int], ln_f: float, ln_q: float, sigma: np.ndarray,
                  n_bootstrap: int, seed: int, stability: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "label": label,
        "ln_f": float(ln_f),
        "ln_Q": float(ln_q),
        "Sigma": [[float(sigma[0, 0]), float(sigma[0, 1])], [float(sigma[1, 0]), float(sigma[1, 1])]],
        "fit": {
            "method": "hilbert_peakband",
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
) -> tuple[dict[str, Any] | None, list[str]]:
    flags: list[str] = []
    try:
        point = estimator(signal, fs)
        ln_f = math.log(float(point["f_hz"]))
        ln_q = math.log(float(point["Q"]))
    except Exception:
        return None, [f"{label}_point_estimate_failed"]

    samples, n_failed = _bootstrap_mode_log_samples(
        signal, fs, estimator, n_bootstrap=n_bootstrap, seed=seed
    )
    total = n_bootstrap
    valid_fraction = float(samples.shape[0] / total) if total > 0 else 0.0
    if samples.shape[0] < 2:
        return None, [f"{label}_bootstrap_insufficient"]

    sigma = compute_covariance(samples)
    ok, reasons = covariance_gate(sigma)
    flags.extend([f"{label}_{r}" for r in reasons])

    if valid_fraction < min_valid_fraction:
        flags.append(f"{label}_valid_fraction_low")
        ok = False

    if cv_threshold is not None:
        f_vals = np.exp(samples[:, 0])
        q_vals = np.exp(samples[:, 1])
        cv_f = float(np.std(f_vals, ddof=1) / (np.mean(f_vals) + 1e-30))
        cv_q = float(np.std(q_vals, ddof=1) / (np.mean(q_vals) + 1e-30))
        if cv_f > cv_threshold:
            flags.append(f"{label}_cv_f_explosive")
            ok = False
        if cv_q > cv_threshold:
            flags.append(f"{label}_cv_Q_explosive")
            ok = False
    else:
        cv_f, cv_q = None, None

    if not (math.isfinite(ln_f) and math.isfinite(ln_q)):
        flags.append(f"{label}_non_finite_log")
        ok = False

    stability = {
        "valid_fraction": valid_fraction,
        "n_successful": int(samples.shape[0]),
        "n_failed": int(n_failed),
        "cv_f": cv_f,
        "cv_Q": cv_q,
    }
    if not ok:
        return None, flags

    return _mode_payload(label, mode, ln_f, ln_q, sigma, n_bootstrap, seed, stability), flags


def build_results_payload(run_id: str, window_meta: dict[str, Any] | None,
                          mode_220: dict[str, Any] | None, mode_221: dict[str, Any] | None,
                          flags: list[str]) -> dict[str, Any]:
    ok = mode_220 is not None and mode_221 is not None
    verdict = "OK" if ok else "INSUFFICIENT_DATA"
    messages = []
    if verdict == "INSUFFICIENT_DATA":
        messages.append("Mode 221 is unstable or missing; stage completed with best-effort estimate.")
    out_modes = []
    if mode_220 is not None:
        out_modes.append(mode_220)
    if mode_221 is not None:
        out_modes.append(mode_221)
    return {
        "schema_version": "multimode_estimates_v1",
        "run_id": run_id,
        "source": {"stage": "s2_ringdown_window", "window": window_meta},
        "modes_target": TARGET_MODES,
        "results": {"verdict": verdict, "quality_flags": sorted(set(flags)), "messages": messages},
        "modes": out_modes,
    }


def _discover_h5_from_s2_manifest(ctx) -> Path:
    manifest_path = ctx.run_dir / "s2_ringdown_window" / "manifest.json"
    check_inputs(ctx, {"s2_manifest": manifest_path})
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Unreadable s2 manifest: {exc}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        abort(ctx, "Corrupt s2 manifest: missing outputs list")

    candidates: list[Path] = []
    for item in outputs:
        path_s = item.get("path") if isinstance(item, dict) else item
        if isinstance(path_s, str) and path_s.endswith(".h5"):
            p = Path(path_s)
            candidates.append(ctx.run_dir / p if not p.is_absolute() else p)

    if not candidates:
        abort(ctx, "No .h5 output found in s2 manifest outputs")

    h5_path = candidates[0]
    if not h5_path.exists() or not h5_path.is_file():
        abort(ctx, f"H5 input not found: {h5_path}")
    return h5_path


def _load_signal_from_h5(path: Path) -> tuple[np.ndarray, float]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"h5py unavailable: {exc}") from exc

    with h5py.File(path, "r") as h5:
        signal = None
        fs = None
        for key in ("strain", "signal", "data"):
            if key in h5:
                arr = np.asarray(h5[key])
                if arr.ndim == 1:
                    signal = arr
                    break
                if arr.ndim >= 2:
                    signal = arr[0]
                    break
        if signal is None:
            for det in ("H1", "L1", "V1"):
                if det in h5:
                    arr = np.asarray(h5[det])
                    signal = arr if arr.ndim == 1 else arr[0]
                    break
        for key in ("sample_rate_hz", "fs"):
            if key in h5:
                fs = float(np.asarray(h5[key]).flat[0])
                break
        if signal is None or fs is None:
            raise RuntimeError("Could not extract signal/fs from h5 schema")

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 16 or not np.all(np.isfinite(signal)):
        raise RuntimeError("Invalid 1D finite signal in h5")
    if not math.isfinite(fs) or fs <= 0:
        raise RuntimeError("Invalid sample rate in h5")
    return signal, fs


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s3b multimode estimates")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ctx = init_stage(args.run_id, STAGE, params={"n_bootstrap": args.n_bootstrap, "seed": args.seed})

    try:
        h5_path = _discover_h5_from_s2_manifest(ctx)
        check_inputs(ctx, {"s2_windowed_h5": h5_path})
        signal, fs = _load_signal_from_h5(h5_path)

        window_meta_path = ctx.run_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"
        window_meta: dict[str, Any] | None = None
        if window_meta_path.exists():
            window_meta = json.loads(window_meta_path.read_text(encoding="utf-8"))

        mode_220, flags_220 = evaluate_mode(
            signal,
            fs,
            label="220",
            mode=[2, 2, 0],
            estimator=_estimate_220,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        mode_221, flags_221 = evaluate_mode(
            signal,
            fs,
            label="221",
            mode=[2, 2, 1],
            estimator=_estimate_221_from_signal,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 1,
            min_valid_fraction=0.8,
            cv_threshold=1.0,
        )

        payload = build_results_payload(args.run_id, window_meta, mode_220, mode_221, flags_220 + flags_221)
        out_path = ctx.outputs_dir / "multimode_estimates.json"
        write_json_atomic(out_path, payload)

        finalize(ctx, {"multimode_estimates": out_path}, results=payload["results"], verdict="PASS")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
