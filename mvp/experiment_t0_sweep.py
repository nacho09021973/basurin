#!/usr/bin/env python3
"""Deterministic t0-sweep experiment over existing s2 ringdown window outputs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

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
from mvp.s3_ringdown_estimates import estimate_ringdown_observables
from mvp.s3b_multimode_estimates import (
    _estimate_220,
    _estimate_221_from_signal,
    evaluate_mode,
)

EXPERIMENT_STAGE = "experiment/t0_sweep"
RESULTS_NAME = "t0_sweep_results.json"


def _parse_grid(args: argparse.Namespace) -> list[int]:
    if args.t0_grid_ms:
        vals = [int(round(float(x.strip()))) for x in args.t0_grid_ms.split(",") if x.strip()]
        if not vals:
            raise ValueError("--t0-grid-ms provided but empty")
        return vals

    if args.t0_start_ms is None or args.t0_stop_ms is None or args.t0_step_ms is None:
        return [0, 5, 10, 15, 20, 25, 30]

    start = int(args.t0_start_ms)
    stop = int(args.t0_stop_ms)
    step = int(args.t0_step_ms)
    if step <= 0:
        raise ValueError("--t0-step-ms must be > 0")
    if stop < start:
        raise ValueError("--t0-stop-ms must be >= --t0-start-ms")
    return list(range(start, stop + 1, step))


def _pick_detector(outputs_dir: Path, detector: str) -> tuple[str, Path]:
    available: dict[str, Path] = {}
    for det in ("H1", "L1", "V1"):
        p = outputs_dir / f"{det}_rd.npz"
        if p.exists():
            available[det] = p

    if not available:
        raise FileNotFoundError(f"No *_rd.npz files in {outputs_dir}")

    if detector != "auto":
        if detector not in available:
            raise FileNotFoundError(f"Requested detector {detector} not found in {outputs_dir}")
        return detector, available[detector]

    for det in ("H1", "L1", "V1"):
        if det in available:
            return det, available[det]
    det = sorted(available.keys())[0]
    return det, available[det]


def _load_npz(npz_path: Path, window_meta: dict[str, Any]) -> tuple[np.ndarray, float]:
    data = np.load(npz_path)
    if "strain" not in data:
        raise RuntimeError(f"corrupt s2 NPZ: missing strain in {npz_path}")
    strain = np.asarray(data["strain"], dtype=float)
    fs = None
    for key in ("sample_rate_hz", "fs"):
        if key in data:
            fs = float(np.asarray(data[key]).flat[0])
            break
    if fs is None:
        for key in ("sample_rate_hz", "fs"):
            if key in window_meta:
                fs = float(window_meta[key])
                break
    if strain.ndim != 1 or strain.size < 16 or not np.all(np.isfinite(strain)):
        raise RuntimeError("corrupt s2 NPZ: invalid strain")
    if fs is None or not math.isfinite(fs) or fs <= 0:
        raise RuntimeError("corrupt s2 NPZ: missing sample_rate_hz")
    return strain, fs


def _run_single_point(signal: np.ndarray, fs: float) -> tuple[dict[str, Any], list[str], str]:
    try:
        est = estimate_ringdown_observables(signal, fs)
    except Exception as exc:
        return {}, [], f"{type(exc).__name__}: {exc}"

    f_hz = float(est["f_hz"])
    q = float(est["Q"])
    ln_f = math.log(f_hz)
    ln_q = math.log(q)
    sigma_logf = float(est.get("sigma_f_hz", 0.0) / f_hz) if f_hz > 0 else 0.0
    sigma_logq = float(est.get("sigma_Q", 0.0) / q) if q > 0 else 0.0
    cov = float(est.get("cov_logf_logQ", 0.0))
    sigma = [[sigma_logf**2, cov], [cov, sigma_logq**2]]
    return {
        "f_hz": f_hz,
        "Q": q,
        "ln_f": ln_f,
        "ln_Q": ln_q,
        "Sigma": sigma,
        "tau_s": float(est.get("tau_s", float("nan"))),
        "snr_peak": float(est.get("snr_peak", float("nan"))),
    }, [], ""


def _run_multimode_point(signal: np.ndarray, fs: float, n_bootstrap: int, seed: int) -> tuple[dict[str, Any], list[str], str]:
    mode_220, flags_220, ok_220 = evaluate_mode(
        signal,
        fs,
        label="220",
        mode=[2, 2, 0],
        estimator=_estimate_220,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    mode_221, flags_221, ok_221 = evaluate_mode(
        signal,
        fs,
        label="221",
        mode=[2, 2, 1],
        estimator=_estimate_221_from_signal,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
        min_valid_fraction=0.8,
        cv_threshold=1.0,
    )
    payload = {
        "verdict": "OK" if (ok_220 and ok_221) else "INSUFFICIENT_DATA",
        "modes": [mode_220, mode_221],
    }
    return payload, sorted(set(flags_220 + flags_221)), ""


def run_t0_sweep(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    out_root = resolve_out_root("runs")
    validate_run_id(args.run_id, out_root)
    require_run_valid(out_root, args.run_id)

    run_dir = out_root / args.run_id
    s2_dir = run_dir / "s2_ringdown_window"
    s2_manifest = s2_dir / "manifest.json"
    if not s2_manifest.exists():
        raise FileNotFoundError(f"Missing s2 manifest: {s2_manifest}")

    s2_outputs = s2_dir / "outputs"
    det, npz_path = _pick_detector(s2_outputs, args.detector)
    window_meta_path = s2_outputs / "window_meta.json"
    window_meta: dict[str, Any] = {}
    if window_meta_path.exists():
        window_meta = json.loads(window_meta_path.read_text(encoding="utf-8"))

    strain, fs = _load_npz(npz_path, window_meta)
    grid = _parse_grid(args)

    stage_dir, outputs_dir = ensure_stage_dirs(args.run_id, EXPERIMENT_STAGE, base_dir=out_root)
    points: list[dict[str, Any]] = []

    n_ok = 0
    n_insufficient = 0
    n_failed = 0
    n_executable = 0

    for t0_ms in grid:
        point: dict[str, Any] = {
            "t0_ms": int(t0_ms),
            "status": "FAILED_POINT",
            "s3": None,
            "s3b": None,
            "quality_flags": [],
            "messages": [],
        }

        if t0_ms < 0:
            point["status"] = "SKIPPED_POINT_EARLY"
            point["messages"].append("t0 offset < 0 is unsupported without re-running s2")
            points.append(point)
            continue

        offset_samples = int(round((float(t0_ms) / 1000.0) * fs))
        if offset_samples >= strain.size:
            point["status"] = "INSUFFICIENT_DATA"
            point["messages"].append("offset exceeds window length")
            n_insufficient += 1
            n_executable += 1
            points.append(point)
            continue

        trimmed = strain[offset_samples:]
        n_executable += 1

        try:
            s3_payload, qflags, err = _run_single_point(trimmed, fs)
            point["s3"] = s3_payload if s3_payload else None
            point["quality_flags"].extend(qflags)
            if err:
                point["status"] = "FAILED_POINT"
                point["messages"].append(err)
                n_failed += 1
                points.append(point)
                continue

            if args.mode == "multimode":
                s3b_payload, flags, mm_err = _run_multimode_point(trimmed, fs, args.n_bootstrap, args.seed)
                point["s3b"] = s3b_payload
                point["quality_flags"].extend(flags)
                if mm_err:
                    point["status"] = "FAILED_POINT"
                    point["messages"].append(mm_err)
                    n_failed += 1
                elif s3b_payload["verdict"] == "INSUFFICIENT_DATA":
                    point["status"] = "INSUFFICIENT_DATA"
                    point["messages"].append("multimode extraction unstable or incomplete")
                    n_insufficient += 1
                else:
                    point["status"] = "OK"
                    n_ok += 1
            else:
                point["status"] = "OK"
                n_ok += 1
        except Exception as exc:
            point["status"] = "FAILED_POINT"
            point["messages"].append(f"{type(exc).__name__}: {exc}")
            n_failed += 1

        points.append(point)

    best_t0 = None
    best_metric = None
    metric_name = "min_sigma_area"
    for point in points:
        if point.get("status") != "OK":
            continue
        s3 = point.get("s3") or {}
        sigma = s3.get("Sigma")
        if not sigma:
            continue
        try:
            arr = np.asarray(sigma, dtype=float)
            metric = float(np.linalg.det(arr))
        except Exception:
            continue
        if not math.isfinite(metric):
            continue
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_t0 = int(point["t0_ms"])

    verdict_note = "EXECUTED"
    if n_executable == 0:
        verdict_note = "SKIPPED_UNSUPPORTED"

    results = {
        "schema_version": "experiment_t0_sweep_v1",
        "run_id": args.run_id,
        "source": {
            "stage": "s2_ringdown_window",
            "detector": det,
            "fs_hz": float(fs),
            "t0_base_s": window_meta.get("t0_start_s"),
            "bandpass_hz": window_meta.get("bandpass_hz"),
        },
        "grid": {
            "t0_offsets_ms": [int(x) for x in grid],
            "interpreted_as": "offsets_from_s2_window_start_nonnegative_only",
        },
        "mode": args.mode,
        "summary": {
            "n_points": len(points),
            "n_ok": int(n_ok),
            "n_insufficient": int(n_insufficient),
            "n_failed": int(n_failed),
            "best_point": {
                "t0_ms": best_t0,
                "metric": metric_name,
                "value": best_metric,
            },
            "verdict": verdict_note,
        },
        "points": points,
    }

    out_path = outputs_dir / RESULTS_NAME
    write_json_atomic(out_path, results)

    stage_summary = {
        "stage": EXPERIMENT_STAGE,
        "run_id": args.run_id,
        "verdict": "PASS",
        "created": utc_now_iso(),
        "results": {
            "n_points": len(points),
            "n_ok": int(n_ok),
            "n_insufficient": int(n_insufficient),
            "n_failed": int(n_failed),
            "experiment_verdict": verdict_note,
        },
        "checks": {
            "run_valid": "PASS",
            "s2_manifest_present": True,
            "s2_npz_present": True,
        },
    }
    write_stage_summary(stage_dir, stage_summary)
    write_manifest(stage_dir, {"t0_sweep_results": out_path})

    return results, stage_summary, out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment: deterministic t0 sweep")
    ap.add_argument("--run-id", "--run", dest="run_id", required=True)
    ap.add_argument("--t0-grid-ms", default=None)
    ap.add_argument("--t0-start-ms", type=int, default=None)
    ap.add_argument("--t0-stop-ms", type=int, default=None)
    ap.add_argument("--t0-step-ms", type=int, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mode", choices=["single", "multimode"], default="single")
    ap.add_argument("--detector", choices=["H1", "L1", "auto"], default="auto")
    ap.add_argument("--atlas-path", default=None)
    args = ap.parse_args()

    try:
        run_t0_sweep(args)
        return 0
    except Exception as exc:
        print(f"[experiment_t0_sweep] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
