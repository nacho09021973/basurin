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
from mvp.s3_ringdown_estimates import estimate_ringdown_observables

STAGE = "s3b_multimode_estimates"
TARGET_MODES = [
    {"mode": [2, 2, 0], "label": "220"},
    {"mode": [2, 2, 1], "label": "221"},
]
MIN_BOOTSTRAP_SAMPLES = 128


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


def _mode_null(label: str, mode: list[int], n_bootstrap: int, seed: int, stability: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "label": label,
        "ln_f": None,
        "ln_Q": None,
        "Sigma": None,
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
    max_lnf_span: float = 1.0,
    max_lnq_span: float = 1.0,
    min_point_samples: int = 2,
    min_point_valid_fraction: float = 0.0,
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
        return _mode_null(label, mode, n_bootstrap, seed, stability), sorted(flags), False

    block_size = _bootstrap_block_size(n_samples)
    if block_size <= 0 or block_size >= n_samples:
        flags.append("bootstrap_block_invalid")
        stability["message"] = "window too short after offset"
        return _mode_null(label, mode, n_bootstrap, seed, stability), sorted(flags), False

    rng = np.random.default_rng(int(seed))

    try:
        samples, n_failed = _bootstrap_mode_log_samples(signal, fs, estimator, n_bootstrap=n_bootstrap, rng=rng)
    except ValueError as exc:
        msg = str(exc)
        if "high <= 0" in msg or "low >= high" in msg:
            flags.append("bootstrap_high_nonpositive")
            stability["message"] = "window too short after offset"
            return _mode_null(label, mode, n_bootstrap, seed, stability), sorted(flags), False
        raise
    sigma_invalid_flag = f"{label}_Sigma_invalid"
    if samples.ndim != 2 or samples.shape[1] != 2:
        flags.append(sigma_invalid_flag)
        return _mode_null(label, mode, n_bootstrap, seed, stability), sorted(set(flags)), False

    valid_mask = np.all(np.isfinite(samples), axis=1)
    dropped_invalid = int(samples.shape[0] - int(np.count_nonzero(valid_mask)))
    if dropped_invalid > 0:
        flags.append(f"{label}_invalid_bootstrap_samples")
    samples = samples[valid_mask]
    n_failed += dropped_invalid

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
        return _mode_null(label, mode, n_bootstrap, seed, stability), sorted(flags), False

    sigma: np.ndarray | None = None
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] < int(min_point_samples):
        flags.append(sigma_invalid_flag)
    else:
        sigma_candidate = compute_covariance(samples)
        if sigma_candidate.shape == (2, 2) and np.all(np.isfinite(sigma_candidate)):
            sigma = sigma_candidate
        else:
            flags.append(sigma_invalid_flag)

    sigma_invertible = False
    if sigma is not None:
        det = float(np.linalg.det(sigma))
        sigma_invertible = math.isfinite(det) and det > 0
        if not sigma_invertible:
            flags.append(sigma_invalid_flag)

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
        mode_payload = _mode_null(label, mode, n_bootstrap, seed, stability)
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
                "method": "hilbert_peakband",
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_seed": int(seed),
                "stability": stability,
            },
        }

    return mode_payload, sorted(set(flags)), ok


def build_results_payload(
    run_id: str,
    window_meta: dict[str, Any] | None,
    mode_220: dict[str, Any],
    mode_220_ok: bool,
    mode_221: dict[str, Any],
    mode_221_ok: bool,
    flags: list[str],
) -> dict[str, Any]:
    verdict = "OK" if (mode_220_ok and mode_221_ok) else "INSUFFICIENT_DATA"
    messages: list[str] = []
    if verdict == "INSUFFICIENT_DATA":
        messages.append("Best-effort multimode fit: one or more modes unavailable or unstable.")
    return {
        "schema_version": "multimode_estimates_v1",
        "run_id": run_id,
        "source": {"stage": "s2_ringdown_window", "window": window_meta},
        "modes_target": TARGET_MODES,
        "results": {
            "verdict": verdict,
            "quality_flags": sorted(set(flags)),
            "messages": sorted(messages),
        },
        "modes": [mode_220, mode_221],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s3b multimode estimates")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).resolve())

    ctx = init_stage(args.run_id, STAGE, params={"n_bootstrap": args.n_bootstrap, "seed": args.seed})

    try:
        window_meta = None
        window_meta_path = _discover_s2_window_meta(ctx.run_dir)
        global_flags: list[str] = []
        if window_meta_path is not None and window_meta_path.exists():
            window_meta = json.loads(window_meta_path.read_text(encoding="utf-8"))
        else:
            global_flags.append("missing_window_meta")
            window_meta_path = ctx.run_dir / "s2_ringdown_window" / "outputs" / "window_meta.json"

        npz_path = _discover_s2_npz(ctx.run_dir)
        check_inputs(ctx, {"s2_rd_npz": npz_path}, optional={"s2_window_meta": window_meta_path})
        signal, fs = _load_signal_from_npz(npz_path, window_meta=window_meta)

        mode_220, flags_220, ok_220 = evaluate_mode(
            signal,
            fs,
            label="220",
            mode=[2, 2, 0],
            estimator=_estimate_220,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            min_point_samples=50,
            min_point_valid_fraction=0.5,
        )
        mode_221, flags_221, ok_221 = evaluate_mode(
            signal,
            fs,
            label="221",
            mode=[2, 2, 1],
            estimator=_estimate_221_from_signal,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 1,
            min_valid_fraction=0.8,
            cv_threshold=1.0,
            min_point_samples=50,
            min_point_valid_fraction=0.5,
        )

        payload = build_results_payload(
            args.run_id,
            window_meta,
            mode_220,
            ok_220,
            mode_221,
            ok_221,
            global_flags + flags_220 + flags_221,
        )

        out_path = ctx.outputs_dir / "multimode_estimates.json"
        write_json_atomic(out_path, payload)
        finalize(ctx, {"multimode_estimates": out_path}, verdict="PASS", results=payload["results"])
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
