#!/usr/bin/env python3
"""Dual-method gate: compare Hilbert vs Lorentzian spectral estimates.

CLI:
    python mvp/experiment_dual_method.py --run <run_id> \
        [--band-low 150] [--band-high 400]

Inputs:
    runs/<run>/s3_ringdown_estimates/outputs/estimates.json        (Hilbert)
    runs/<run>/s3_spectral_estimates/outputs/spectral_estimates.json (Spectral)
    (runs s3 and/or s3_spectral if outputs not already present)

Outputs:
    runs/<run>/experiment/DUAL_METHOD_V1/dual_method_comparison.json

Verdict logic:
    tension_f = |Δf| / sqrt(σ_f_h² + σ_f_s²)
    tension_Q = |ΔQ| / sqrt(σ_Q_h² + σ_Q_s²)

    CONSISTENT   if max(tension_f, tension_Q) < 2
    TENSION      if 2 ≤ max(tension_f, tension_Q) < 3
    INCONSISTENT if max(tension_f, tension_Q) ≥ 3
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
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

MVP_DIR = Path(__file__).resolve().parent


def _run_stage(script: str, args: list[str]) -> int:
    """Run an MVP stage script as a subprocess."""
    cmd = [sys.executable, str(MVP_DIR / script)] + args
    print(f"[dual_method] Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_estimates(data: dict[str, Any]) -> dict[str, float]:
    """Extract (f_hz, Q, tau_s, sigma_f_hz, sigma_Q) from estimates JSON."""
    combined = data.get("combined", {})
    unc = data.get("combined_uncertainty", {})

    f_hz = float(combined.get("f_hz", float("nan")))
    Q = float(combined.get("Q", float("nan")))
    tau_s = float(combined.get("tau_s", float("nan")))

    # Prefer combined_uncertainty, then combined fields
    sigma_f = unc.get("sigma_f_hz") or combined.get("sigma_f_hz") or float("nan")
    sigma_Q = unc.get("sigma_Q") or combined.get("sigma_Q") or float("nan")

    return {
        "f_hz": f_hz, "Q": Q, "tau_s": tau_s,
        "sigma_f_hz": float(sigma_f) if sigma_f is not None else float("nan"),
        "sigma_Q": float(sigma_Q) if sigma_Q is not None else float("nan"),
    }


def compare_methods(
    hilbert: dict[str, float],
    spectral: dict[str, float],
) -> dict[str, Any]:
    """Compute tension between Hilbert and spectral estimates."""
    delta_f = abs(hilbert["f_hz"] - spectral["f_hz"])
    delta_Q = abs(hilbert["Q"] - spectral["Q"])

    sigma_f_combined = math.sqrt(
        hilbert["sigma_f_hz"] ** 2 + spectral["sigma_f_hz"] ** 2
    )
    sigma_Q_combined = math.sqrt(
        hilbert["sigma_Q"] ** 2 + spectral["sigma_Q"] ** 2
    )

    if sigma_f_combined > 0:
        tension_f = delta_f / sigma_f_combined
    else:
        tension_f = float("inf")

    if sigma_Q_combined > 0:
        tension_Q = delta_Q / sigma_Q_combined
    else:
        tension_Q = float("inf")

    max_tension = max(tension_f, tension_Q)
    if max_tension < 2.0:
        verdict = "CONSISTENT"
    elif max_tension < 3.0:
        verdict = "TENSION"
    else:
        verdict = "INCONSISTENT"

    return {
        "delta_f_hz": delta_f,
        "delta_Q": delta_Q,
        "sigma_f_combined": sigma_f_combined,
        "sigma_Q_combined": sigma_Q_combined,
        "tension_f_sigma": tension_f,
        "tension_Q_sigma": tension_Q,
        "verdict": verdict,
    }


def run_dual_method(
    run_id: str,
    band_low: float = 150.0,
    band_high: float = 400.0,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Run dual-method comparison for a given run."""
    if out_root is None:
        out_root = resolve_out_root("runs")

    run_dir = out_root / run_id

    hilbert_path = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    spectral_path = run_dir / "s3_spectral_estimates" / "outputs" / "spectral_estimates.json"

    # Run s3 Hilbert if not already done
    if not hilbert_path.exists():
        print("[dual_method] Running s3_ringdown_estimates ...", flush=True)
        rc = _run_stage("s3_ringdown_estimates.py", [
            "--run", run_id,
            "--band-low", str(band_low),
            "--band-high", str(band_high),
        ])
        if rc != 0:
            raise RuntimeError(f"s3_ringdown_estimates failed with exit code {rc}")

    # Run s3 spectral if not already done
    if not spectral_path.exists():
        print("[dual_method] Running s3_spectral_estimates ...", flush=True)
        rc = _run_stage("s3_spectral_estimates.py", [
            "--run", run_id,
            "--band-low", str(band_low),
            "--band-high", str(band_high),
        ])
        if rc != 0:
            raise RuntimeError(f"s3_spectral_estimates failed with exit code {rc}")

    # Load both
    hilbert_data = _load_json(hilbert_path)
    spectral_data = _load_json(spectral_path)

    hilbert_est = _extract_estimates(hilbert_data)
    spectral_est = _extract_estimates(spectral_data)

    # Check if spectral converged
    spectral_converged = True
    per_det = spectral_data.get("per_detector", {})
    if all(not d.get("fit_converged", True) for d in per_det.values()
           if isinstance(d, dict)):
        spectral_converged = False

    # Also check for NaN in spectral estimates
    if not math.isfinite(spectral_est["f_hz"]) or not math.isfinite(spectral_est["Q"]):
        spectral_converged = False

    # Compare
    if spectral_converged and math.isfinite(hilbert_est["f_hz"]):
        comparison = compare_methods(hilbert_est, spectral_est)
    else:
        comparison = {
            "delta_f_hz": float("nan"),
            "delta_Q": float("nan"),
            "sigma_f_combined": float("nan"),
            "sigma_Q_combined": float("nan"),
            "tension_f_sigma": float("nan"),
            "tension_Q_sigma": float("nan"),
            "verdict": "INCONSISTENT",
            "note": "spectral fit did not converge",
        }

    # Recommendation
    if not spectral_converged:
        recommendation = "hilbert"
        recommended_est = hilbert_est
    else:
        recommendation = "spectral"
        recommended_est = spectral_est

    result: dict[str, Any] = {
        "schema_version": "mvp_dual_method_v1",
        "run_id": run_id,
        "event_id": hilbert_data.get("event_id", "unknown"),
        "band_hz": [band_low, band_high],
        "created_utc": utc_now_iso(),
        "hilbert": hilbert_est,
        "spectral": spectral_est,
        "spectral_converged": spectral_converged,
        "comparison": comparison,
        "recommendation": recommendation,
        "recommended_estimates": recommended_est,
    }

    # Write output
    exp_dir = run_dir / "experiment" / "DUAL_METHOD_V1"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_path = exp_dir / "dual_method_comparison.json"
    write_json_atomic(out_path, result)

    # Write manifest
    manifest = {
        "schema_version": "mvp_manifest_v1",
        "stage": "experiment_dual_method",
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "artifacts": [
            {
                "path": str(out_path.relative_to(run_dir)),
                "sha256": sha256_file(out_path),
            }
        ],
        "inputs": [
            {"path": str(hilbert_path.relative_to(run_dir)), "label": "hilbert_estimates"},
            {"path": str(spectral_path.relative_to(run_dir)), "label": "spectral_estimates"},
        ],
    }
    write_json_atomic(exp_dir / "manifest.json", manifest)

    print(f"[dual_method] Verdict: {comparison['verdict']}", flush=True)
    print(f"[dual_method] Recommendation: {recommendation}", flush=True)
    print(f"[dual_method] Output: {out_path}", flush=True)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Dual-method gate: compare Hilbert vs spectral estimates"
    )
    ap.add_argument("--run", required=True, help="Run ID")
    ap.add_argument("--band-low", type=float, default=150.0)
    ap.add_argument("--band-high", type=float, default=400.0)
    args = ap.parse_args()

    try:
        run_dual_method(args.run, args.band_low, args.band_high)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
