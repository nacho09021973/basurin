#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from basurin_io import (
    require_run_valid,
    resolve_out_root,
    resolve_spectrum_path,
    sha256_file,
    write_manifest,
    write_stage_summary,
)


STAGE_NAME = "experiment/uldm_laser_00"


def abort(msg: str) -> None:
    raise SystemExit(2, f"[BASURIN ABORT] {msg}")


def _read_spectrum_frequencies(path: Path, n_freqs: int, fs: float) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "delta_uv" in h5:
            vals = np.asarray(h5["delta_uv"][:], dtype=float).ravel()
        else:
            vals = None
            for key in ("masses", "M2", "eigenvalues"):
                if key in h5:
                    vals = np.asarray(h5[key][:], dtype=float).ravel()
                    break
    if vals is None or vals.size == 0:
        return np.linspace(2_000.0, min(20_000.0, fs / 3.0), n_freqs, dtype=float)

    vals = np.abs(vals[np.isfinite(vals)])
    if vals.size == 0:
        return np.linspace(2_000.0, min(20_000.0, fs / 3.0), n_freqs, dtype=float)

    vmax = np.percentile(vals, 95)
    if vmax <= 0:
        vmax = 1.0
    scaled = (vals / vmax) * min(20_000.0, fs / 3.0)
    scaled = np.clip(scaled, 200.0, min(20_000.0, fs / 3.0))
    uniq = np.unique(np.round(scaled, 3))
    if uniq.size >= n_freqs:
        idx = np.linspace(0, uniq.size - 1, n_freqs).astype(int)
        return uniq[idx]
    fill = np.linspace(200.0, min(20_000.0, fs / 3.0), n_freqs)
    out = np.unique(np.concatenate([uniq, fill]))
    return out[:n_freqs]


def _msc_at_frequency(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    f0: float,
    nfft: int,
) -> float:
    step = nfft // 2
    if x.size < nfft:
        return 0.0
    w = np.hanning(nfft)
    pxx = 0.0
    pyy = 0.0
    pxy = 0.0j
    count = 0
    for i in range(0, x.size - nfft + 1, step):
        sx = np.fft.rfft(x[i : i + nfft] * w)
        sy = np.fft.rfft(y[i : i + nfft] * w)
        pxx += sx * np.conj(sx)
        pyy += sy * np.conj(sy)
        pxy += sx * np.conj(sy)
        count += 1
    if count == 0:
        return 0.0
    pxx /= count
    pyy /= count
    pxy /= count
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - f0)))
    num = np.abs(pxy[idx]) ** 2
    den = np.real(pxx[idx]) * np.real(pyy[idx]) + 1e-18
    return float(np.clip(num / den, 0.0, 1.0))


def _simulate_stat(
    rng: np.random.Generator,
    f0: float,
    fs: float,
    n_samples: int,
    nfft: int,
    noise_sigma: float,
    amp: float,
    g_strength: float,
    phase_shift: float,
) -> float:
    t = np.arange(n_samples, dtype=float) / fs
    phi = rng.uniform(0.0, 2.0 * np.pi)
    n1 = rng.normal(0.0, noise_sigma, size=n_samples)
    n2 = rng.normal(0.0, noise_sigma, size=n_samples)
    g = 1.0 + g_strength * (f0 / min(20_000.0, fs / 3.0))
    s1 = amp * np.sin(2.0 * np.pi * f0 * t + phi)
    s2 = amp * g * np.sin(2.0 * np.pi * f0 * t + phi + phase_shift)
    return _msc_at_frequency(n1 + s1, n2 + s2, fs, f0, nfft)


def run_experiment(args: argparse.Namespace) -> int:
    out_root = resolve_out_root(args.out_root)
    run_dir = out_root / args.run

    try:
        _ = require_run_valid(out_root, args.run)
    except RuntimeError as exc:
        raise SystemExit(2) from exc

    spectrum_path = resolve_spectrum_path(run_dir)
    if not spectrum_path.exists():
        raise SystemExit(2)

    stage_dir = run_dir / STAGE_NAME
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    freqs = _read_spectrum_frequencies(spectrum_path, args.n_freqs, args.fs)
    rng = np.random.default_rng(args.seed)
    n_samples = int(args.fs * args.duration_s)

    null_stats = []
    inj_stats = []
    for _ in range(args.n_trials):
        f0 = float(freqs[rng.integers(0, freqs.size)])
        null_stats.append(
            _simulate_stat(
                rng,
                f0,
                args.fs,
                n_samples,
                args.nfft,
                args.noise_sigma,
                args.a_null,
                args.g_strength,
                args.phase_shift,
            )
        )
        inj_stats.append(
            _simulate_stat(
                rng,
                f0,
                args.fs,
                n_samples,
                args.nfft,
                args.noise_sigma,
                args.a_inj,
                args.g_strength,
                args.phase_shift,
            )
        )

    null_arr = np.asarray(null_stats, dtype=float)
    inj_arr = np.asarray(inj_stats, dtype=float)
    threshold = float(np.quantile(null_arr, 0.99))
    tpr = float(np.mean(inj_arr > threshold))

    stats: dict[str, Any] = {
        "null": {
            "p50": float(np.percentile(null_arr, 50)),
            "p90": float(np.percentile(null_arr, 90)),
            "p99": float(np.percentile(null_arr, 99)),
        },
        "inj": {
            "p50": float(np.percentile(inj_arr, 50)),
            "p90": float(np.percentile(inj_arr, 90)),
            "p99": float(np.percentile(inj_arr, 99)),
            "tpr_at_fpr_1pct": tpr,
        },
        "thresholds": {
            "fpr": 0.01,
            "threshold_from_null": threshold,
            "threshold_null_max": args.threshold_null_max,
            "threshold_tpr_min": args.threshold_tpr_min,
        },
        "seed": args.seed,
    }
    injections = {
        "a_null": args.a_null,
        "a_inj": args.a_inj,
        "noise_sigma": args.noise_sigma,
        "g_strength": args.g_strength,
        "phase_shift": args.phase_shift,
        "fs": args.fs,
        "duration_s": args.duration_s,
        "nfft": args.nfft,
        "n_trials": args.n_trials,
        "frequencies_hz": [float(x) for x in freqs.tolist()],
    }

    verdict = "PASS"
    reasons = []
    if stats["null"]["p99"] >= args.threshold_null_max:
        verdict = "FAIL"
        reasons.append("null_p99_above_limit")
    if tpr <= args.threshold_tpr_min:
        verdict = "FAIL"
        reasons.append("tpr_below_min")

    stats_path = outputs_dir / "stats.json"
    injections_path = outputs_dir / "injections.json"
    traces_path = outputs_dir / "traces.npz"

    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    injections_path.write_text(json.dumps(injections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez(
        traces_path,
        null_stats=null_arr,
        inj_stats=inj_arr,
        frequencies_hz=freqs,
    )

    summary = {
        "stage": STAGE_NAME,
        "run": args.run,
        "inputs": {
            "spectrum_path": str(spectrum_path),
            "spectrum_sha256": sha256_file(spectrum_path),
        },
        "parameters": injections,
        "verdict": verdict,
        "reason": ",".join(reasons) if reasons else "ok",
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "stats": stats_path,
            "injections": injections_path,
            "traces": traces_path,
            "stage_summary": summary_path,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXP ULDM laser coherence gate")
    p.add_argument("--run", required=True)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fs", type=float, default=100_000.0)
    p.add_argument("--duration-s", type=float, default=0.2)
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--n-freqs", type=int, default=16)
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--noise-sigma", type=float, default=1.0)
    p.add_argument("--a-null", type=float, default=0.0)
    p.add_argument("--a-inj", type=float, default=3.0)
    p.add_argument("--g-strength", type=float, default=0.08)
    p.add_argument("--phase-shift", type=float, default=0.03)
    p.add_argument("--threshold-null-max", type=float, default=0.35)
    p.add_argument("--threshold-tpr-min", type=float, default=0.9)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
