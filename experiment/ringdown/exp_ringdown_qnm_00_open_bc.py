#!/usr/bin/env python3
"""EXP_RINGDOWN_QNM_00 — Open Boundary Condition experiment.

Introduces "horizon-like" absorbing boundary conditions to produce complex
eigenfrequencies (QNMs) with decay (Im omega < 0).

Contracts:
  C1 — Horizon-like decay: omega_I < -eps
  C2 — Resonance stability: omega_R/omega_I stable under grid/window sweeps

This is an experimental stage; it does not contaminate canonical Bloque B.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2

# Default tolerances for contracts
DEFAULT_DECAY_EPS = 1e-8
DEFAULT_OMEGA_R_REL_TOL = 0.05  # 5% relative tolerance for omega_R stability
DEFAULT_OMEGA_I_REL_TOL = 0.10  # 10% relative tolerance for omega_I stability
DEFAULT_FIT_R2_MIN = 0.90       # minimum R^2 for fit quality


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(p: Path) -> dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_run_valid(run_dir: Path) -> dict[str, Any]:
    """Check RUN_VALID gate is PASS."""
    for p in [
        run_dir / "RUN_VALID" / "verdict.json",
        run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
    ]:
        if p.exists():
            v = _read_json(p)
            verdict = str(v.get("verdict", v.get("status", ""))).upper()
            if verdict != "PASS":
                raise SystemExit(f"ERROR: RUN_VALID={verdict} at {p}")
            return v
    raise SystemExit(f"ERROR: RUN_VALID not found in {run_dir}")


def _parse_grid_sweep(spec: str) -> list[int]:
    """Parse grid sweep spec like 'N=1024,2048,4096' into list of ints."""
    if spec.startswith("N="):
        spec = spec[2:]
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_window_sweep(spec: str) -> list[str]:
    """Parse window sweep spec like 'w1,w2,w3' into list of window IDs."""
    return [x.strip() for x in spec.split(",") if x.strip()]


def _generate_synthetic_damped_signal(
    f_hz: float,
    tau_s: float,
    duration: float = 1.0,
    sample_rate: float = 4096.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic damped sinusoid (ringdown-like)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / sample_rate)
    # h(t) = A * exp(-t/tau) * cos(2*pi*f*t + phi)
    phi = rng.uniform(0, 2 * np.pi)
    amplitude = 1.0
    signal = amplitude * np.exp(-t / tau_s) * np.cos(2 * np.pi * f_hz * t + phi)
    # Add small noise
    noise_level = 0.01 * amplitude
    signal += rng.normal(0, noise_level, size=len(t))
    return t, signal


def _fit_damped_sinusoid(
    t: np.ndarray,
    signal: np.ndarray,
    f_guess: float | None = None,
    tau_guess: float | None = None,
) -> dict[str, Any]:
    """Fit damped sinusoid to extract omega_R (frequency) and omega_I (decay rate).

    Model: h(t) = A * exp(omega_I * t) * cos(omega_R * t + phi)
    where omega_I < 0 for decay.

    This is a simplified fit using FFT + exponential envelope for demonstration.
    A production version would use proper nonlinear least squares.
    """
    # FFT to estimate frequency
    dt = t[1] - t[0]
    n = len(signal)
    freqs = np.fft.rfftfreq(n, dt)
    fft_mag = np.abs(np.fft.rfft(signal))

    # Find dominant frequency
    peak_idx = np.argmax(fft_mag[1:]) + 1  # skip DC
    omega_R = 2 * np.pi * freqs[peak_idx]

    # Estimate decay from envelope (using Hilbert transform approximation)
    # |h(t)| ~ A * exp(omega_I * t)
    analytic = np.abs(signal + 1j * np.imag(np.fft.ifft(np.fft.fft(signal) * (1 - np.sign(np.fft.fftfreq(n))))))
    envelope = np.maximum(analytic, 1e-12)

    # Linear fit on log envelope to get decay rate
    log_env = np.log(envelope + 1e-12)
    # Use only first half to avoid noise floor
    n_fit = n // 2
    if n_fit < 10:
        n_fit = n
    coeffs = np.polyfit(t[:n_fit], log_env[:n_fit], 1)
    omega_I = coeffs[0]  # slope = omega_I (should be negative for decay)

    # Compute fit quality (R^2)
    y_pred = np.exp(coeffs[0] * t[:n_fit] + coeffs[1])
    ss_res = np.sum((envelope[:n_fit] - y_pred) ** 2)
    ss_tot = np.sum((envelope[:n_fit] - np.mean(envelope[:n_fit])) ** 2)
    fit_r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    # Residual norm
    residual_norm = np.sqrt(ss_res / n_fit)

    return {
        "omega_R": float(omega_R),
        "omega_I": float(omega_I),
        "omega_complex": [float(omega_R), float(omega_I)],
        "f_hz": float(omega_R / (2 * np.pi)),
        "tau_s": float(-1.0 / omega_I) if omega_I < 0 else float("inf"),
        "fit_r2": float(fit_r2),
        "residual_norm": float(residual_norm),
        "n_samples": int(n),
    }


def _run_single_case(
    case_id: str,
    t: np.ndarray,
    signal: np.ndarray,
    n_grid: int,
    window_id: str,
) -> dict[str, Any]:
    """Run QNM fit for a single grid/window configuration."""
    # Resample to specified grid size
    n_orig = len(t)
    if n_grid < n_orig:
        indices = np.linspace(0, n_orig - 1, n_grid, dtype=int)
        t_resampled = t[indices]
        signal_resampled = signal[indices]
    else:
        t_resampled = t
        signal_resampled = signal

    # Apply window (simple windowing for demonstration)
    if window_id == "w1":  # no window
        window = np.ones_like(signal_resampled)
    elif window_id == "w2":  # Hann window
        window = np.hanning(len(signal_resampled))
    elif window_id == "w3":  # Tukey window (0.5 alpha)
        from scipy.signal import windows
        window = windows.tukey(len(signal_resampled), alpha=0.5)
    else:
        window = np.ones_like(signal_resampled)

    signal_windowed = signal_resampled * window

    fit_result = _fit_damped_sinusoid(t_resampled, signal_windowed)

    return {
        "case_id": case_id,
        "n_grid": n_grid,
        "window_id": window_id,
        "omega_R": fit_result["omega_R"],
        "omega_I": fit_result["omega_I"],
        "omega_complex": fit_result["omega_complex"],
        "f_hz": fit_result["f_hz"],
        "tau_s": fit_result["tau_s"],
        "fit_r2": fit_result["fit_r2"],
        "residual_norm": fit_result["residual_norm"],
    }


def _evaluate_contract_c1(
    cases: list[dict[str, Any]],
    decay_eps: float,
    fit_r2_min: float,
) -> dict[str, Any]:
    """Contract C1: Horizon-like decay.

    PASS if:
      - omega_I < -eps for all cases
      - fit_r2 >= fit_r2_min for all cases
    """
    violations: list[dict[str, Any]] = []

    for case in cases:
        omega_I = case["omega_I"]
        fit_r2 = case["fit_r2"]

        if omega_I >= -decay_eps:
            violations.append({
                "case_id": case["case_id"],
                "metric": "omega_I",
                "value": omega_I,
                "threshold": -decay_eps,
                "reason": "omega_I not sufficiently negative (no decay)",
            })

        if fit_r2 < fit_r2_min:
            violations.append({
                "case_id": case["case_id"],
                "metric": "fit_r2",
                "value": fit_r2,
                "threshold": fit_r2_min,
                "reason": "fit quality below threshold",
            })

    return {
        "contract": "C1_horizon_decay",
        "verdict": "PASS" if len(violations) == 0 else "FAIL",
        "thresholds": {
            "decay_eps": decay_eps,
            "fit_r2_min": fit_r2_min,
        },
        "violations": violations,
    }


def _evaluate_contract_c2(
    cases: list[dict[str, Any]],
    omega_R_rel_tol: float,
    omega_I_rel_tol: float,
) -> dict[str, Any]:
    """Contract C2: Resonance stability under grid/window variations.

    PASS if:
      - omega_R varies < omega_R_rel_tol across all cases
      - omega_I varies < omega_I_rel_tol across all cases
    """
    if len(cases) < 2:
        return {
            "contract": "C2_stability",
            "verdict": "PASS",
            "reason": "single case, stability not applicable",
            "thresholds": {
                "omega_R_rel_tol": omega_R_rel_tol,
                "omega_I_rel_tol": omega_I_rel_tol,
            },
            "violations": [],
        }

    omega_Rs = np.array([c["omega_R"] for c in cases])
    omega_Is = np.array([c["omega_I"] for c in cases])

    # Compute statistics
    omega_R_mean = np.mean(omega_Rs)
    omega_I_mean = np.mean(omega_Is)

    omega_R_std = np.std(omega_Rs)
    omega_I_std = np.std(omega_Is)

    omega_R_rel_var = omega_R_std / (np.abs(omega_R_mean) + 1e-12)
    omega_I_rel_var = omega_I_std / (np.abs(omega_I_mean) + 1e-12)

    # Percentiles
    omega_R_p50 = float(np.percentile(omega_Rs, 50))
    omega_R_p90 = float(np.percentile(np.abs(omega_Rs - omega_R_mean), 90))
    omega_I_p50 = float(np.percentile(omega_Is, 50))
    omega_I_p90 = float(np.percentile(np.abs(omega_Is - omega_I_mean), 90))

    violations: list[dict[str, Any]] = []

    if omega_R_rel_var > omega_R_rel_tol:
        violations.append({
            "metric": "omega_R",
            "rel_var": float(omega_R_rel_var),
            "threshold": omega_R_rel_tol,
            "reason": "omega_R variation exceeds tolerance",
        })

    if omega_I_rel_var > omega_I_rel_tol:
        violations.append({
            "metric": "omega_I",
            "rel_var": float(omega_I_rel_var),
            "threshold": omega_I_rel_tol,
            "reason": "omega_I variation exceeds tolerance",
        })

    return {
        "contract": "C2_stability",
        "verdict": "PASS" if len(violations) == 0 else "FAIL",
        "thresholds": {
            "omega_R_rel_tol": omega_R_rel_tol,
            "omega_I_rel_tol": omega_I_rel_tol,
        },
        "statistics": {
            "omega_R": {
                "mean": float(omega_R_mean),
                "std": float(omega_R_std),
                "rel_var": float(omega_R_rel_var),
                "p50": omega_R_p50,
                "p90_deviation": omega_R_p90,
            },
            "omega_I": {
                "mean": float(omega_I_mean),
                "std": float(omega_I_std),
                "rel_var": float(omega_I_rel_var),
                "p50": omega_I_p50,
                "p90_deviation": omega_I_p90,
            },
        },
        "violations": violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EXP_RINGDOWN_QNM_00 — Open Boundary Condition experiment."
    )
    ap.add_argument("--run", required=True, help="Run ID")
    ap.add_argument("--out-root", default="runs", help="Output root directory")
    ap.add_argument(
        "--in-spectrum",
        help="Path to upstream spectrum.h5 (optional, for metadata traceability)",
    )
    ap.add_argument(
        "--config",
        help="Path to JSON config file with open BC parameters",
    )
    ap.add_argument(
        "--model",
        default="open_bc_resonance",
        help="Model type for open BC fit",
    )
    ap.add_argument(
        "--grid-sweep",
        default="N=1024,2048,4096",
        help="Grid sizes to sweep (e.g., 'N=1024,2048,4096')",
    )
    ap.add_argument(
        "--window-sweep",
        default="w1,w2,w3",
        help="Window functions to sweep (e.g., 'w1,w2,w3')",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Tolerance overrides
    ap.add_argument("--decay-eps", type=float, default=DEFAULT_DECAY_EPS)
    ap.add_argument("--omega-r-rel-tol", type=float, default=DEFAULT_OMEGA_R_REL_TOL)
    ap.add_argument("--omega-i-rel-tol", type=float, default=DEFAULT_OMEGA_I_REL_TOL)
    ap.add_argument("--fit-r2-min", type=float, default=DEFAULT_FIT_R2_MIN)

    # Synthetic signal parameters (for surrogate mode)
    ap.add_argument("--f-hz", type=float, default=250.0, help="Frequency in Hz")
    ap.add_argument("--tau-s", type=float, default=0.004, help="Decay time in seconds")

    args = ap.parse_args()

    # Validate run
    try:
        out_root = resolve_out_root(args.out_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_dir = get_run_dir(args.run, base_dir=out_root)

    # Check RUN_VALID if exists (optional for experimental stages)
    run_valid_payload = None
    try:
        run_valid_payload = _require_run_valid(run_dir)
    except SystemExit:
        # RUN_VALID not required for experimental stages, but log it
        print("WARNING: RUN_VALID not found or not PASS, continuing anyway", file=sys.stderr)

    # Create stage directory
    stage_name = "experiment/ringdown/EXP_RINGDOWN_QNM_00_open_bc"
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, stage_name, base_dir=out_root)

    # Parse sweep specifications
    grid_sizes = _parse_grid_sweep(args.grid_sweep)
    window_ids = _parse_window_sweep(args.window_sweep)

    # Load config if provided
    config: dict[str, Any] = {}
    config_hash = None
    if args.config and Path(args.config).exists():
        config = _read_json(Path(args.config))
        config_hash = sha256_file(Path(args.config))

    # Load or generate synthetic signal
    # For now, we use a surrogate synthetic signal
    # In production, this would load from strain/PSD or spectrum.h5
    t, signal = _generate_synthetic_damped_signal(
        f_hz=args.f_hz,
        tau_s=args.tau_s,
        duration=1.0,
        sample_rate=4096.0,
        seed=args.seed,
    )

    # Truth values for reference
    truth = {
        "f_hz": args.f_hz,
        "tau_s": args.tau_s,
        "omega_R_true": 2 * np.pi * args.f_hz,
        "omega_I_true": -1.0 / args.tau_s,
    }

    # Run sweep
    cases: list[dict[str, Any]] = []
    case_idx = 0
    for n_grid in grid_sizes:
        for window_id in window_ids:
            case_id = f"case_{case_idx:03d}"
            result = _run_single_case(case_id, t, signal, n_grid, window_id)
            result["truth"] = truth
            cases.append(result)
            case_idx += 1

    # Write per-case results
    per_case_dir = outputs_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        case_path = per_case_dir / f"{case['case_id']}.json"
        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2)

    # Evaluate contracts
    c1_result = _evaluate_contract_c1(
        cases,
        decay_eps=args.decay_eps,
        fit_r2_min=args.fit_r2_min,
    )
    c2_result = _evaluate_contract_c2(
        cases,
        omega_R_rel_tol=args.omega_r_rel_tol,
        omega_I_rel_tol=args.omega_i_rel_tol,
    )

    # Overall verdict
    overall_verdict = "PASS" if (c1_result["verdict"] == "PASS" and c2_result["verdict"] == "PASS") else "FAIL"

    # Build qnm_fit.json (canonical output)
    # Use first case as reference (or mean across cases)
    omega_Rs = [c["omega_R"] for c in cases]
    omega_Is = [c["omega_I"] for c in cases]

    qnm_fit = {
        "schema_version": "qnm_fit_v1",
        "created": utc_now_iso(),
        "model": args.model,
        "omega_complex": [float(np.mean(omega_Rs)), float(np.mean(omega_Is))],
        "omega_R": float(np.mean(omega_Rs)),
        "omega_I": float(np.mean(omega_Is)),
        "omega_R_std": float(np.std(omega_Rs)),
        "omega_I_std": float(np.std(omega_Is)),
        "f_hz": float(np.mean(omega_Rs) / (2 * np.pi)),
        "tau_s": float(-1.0 / np.mean(omega_Is)) if np.mean(omega_Is) < 0 else None,
        "n_cases": len(cases),
        "grid_sizes": grid_sizes,
        "window_ids": window_ids,
        "truth": truth,
        "stability_metrics": c2_result.get("statistics", {}),
    }

    qnm_fit_path = outputs_dir / "qnm_fit.json"
    with open(qnm_fit_path, "w", encoding="utf-8") as f:
        json.dump(qnm_fit, f, indent=2)

    # Build contract_verdict.json
    contract_verdict = {
        "schema_version": "contract_verdict_v1",
        "created": utc_now_iso(),
        "verdict": overall_verdict,
        "contracts": {
            "C1_horizon_decay": c1_result,
            "C2_stability": c2_result,
        },
        "assumptions": [
            "Surrogate synthetic signal (damped sinusoid) used for demonstration",
            "Production version should use actual strain/PSD or QNM solver output",
            "omega_I < 0 indicates exponential decay (horizon-like absorption)",
        ],
        "inputs": {},
    }

    # Add input references if available
    if args.in_spectrum and Path(args.in_spectrum).exists():
        contract_verdict["inputs"]["spectrum"] = {
            "path": args.in_spectrum,
            "sha256": sha256_file(Path(args.in_spectrum)),
        }
    if args.config and Path(args.config).exists():
        contract_verdict["inputs"]["config"] = {
            "path": args.config,
            "sha256": config_hash,
        }
    if run_valid_payload:
        rv_path = run_dir / "RUN_VALID" / "verdict.json"
        if rv_path.exists():
            contract_verdict["inputs"]["RUN_VALID"] = {
                "path": str(rv_path),
                "sha256": sha256_file(rv_path),
            }

    contract_verdict_path = outputs_dir / "contract_verdict.json"
    with open(contract_verdict_path, "w", encoding="utf-8") as f:
        json.dump(contract_verdict, f, indent=2)

    # Write stage_summary
    summary = {
        "stage": stage_name,
        "script": "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
        "params": {
            "model": args.model,
            "grid_sweep": args.grid_sweep,
            "window_sweep": args.window_sweep,
            "seed": args.seed,
            "f_hz": args.f_hz,
            "tau_s": args.tau_s,
            "decay_eps": args.decay_eps,
            "omega_r_rel_tol": args.omega_r_rel_tol,
            "omega_i_rel_tol": args.omega_i_rel_tol,
            "fit_r2_min": args.fit_r2_min,
        },
        "inputs": contract_verdict["inputs"],
        "outputs": {
            "qnm_fit": "outputs/qnm_fit.json",
            "contract_verdict": "outputs/contract_verdict.json",
            "per_case_dir": "outputs/per_case/",
        },
        "verdict": overall_verdict,
    }
    write_stage_summary(stage_dir, summary)

    # Write manifest
    artifacts = {
        "qnm_fit": qnm_fit_path,
        "contract_verdict": contract_verdict_path,
    }
    for p in sorted(per_case_dir.glob("case_*.json")):
        artifacts[f"per_case::{p.name}"] = p

    write_manifest(stage_dir, artifacts, extra={"verdict": overall_verdict})

    print(f"Stage completed: {stage_name}")
    print(f"  Verdict: {overall_verdict}")
    print(f"  omega_R = {qnm_fit['omega_R']:.6f} (true: {truth['omega_R_true']:.6f})")
    print(f"  omega_I = {qnm_fit['omega_I']:.6f} (true: {truth['omega_I_true']:.6f})")
    print(f"  C1 (decay): {c1_result['verdict']}")
    print(f"  C2 (stability): {c2_result['verdict']}")

    if overall_verdict != "PASS":
        return EXIT_CONTRACT_FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
