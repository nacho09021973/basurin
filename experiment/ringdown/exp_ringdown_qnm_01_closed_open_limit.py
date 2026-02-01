#!/usr/bin/env python3
"""EXP_RINGDOWN_QNM_01 — Closed ↔ Open limit comparison.

Verifies internal consistency: when absorption → 0, the open-boundary
model (QNM with complex ω) should recover the closed-boundary spectrum
(Bloque B with real M²).

Physical interpretation:
  - Closed (Bloque B): "nail" boundary, Hermitian operator, real eigenvalues
  - Open (QNM): "horizon" absorber, non-Hermitian, complex eigenvalues
  - Limit: as absorption → 0, Im(ω) → 0 and Re(ω)² → M²

Contracts:
  C3 — Closed limit recovery: |ω_R² - M²| < tol when absorption → 0
  C4 — Monotonicity: |ω_I| increases monotonically with absorption

This validates internal pipeline consistency before connecting to real GW data.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[1], _here.parents[2], _here.parents[3]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# --------------------------------

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    resolve_out_root,
    resolve_spectrum_path,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2

# Default tolerances
DEFAULT_OMEGA_R_SQ_REL_TOL = 0.05  # 5% relative error for ω_R² vs M²
DEFAULT_OMEGA_I_ZERO_TOL = 1e-6    # absolute tolerance for ω_I → 0
DEFAULT_MONOTONICITY_TOL = 0.0     # strict monotonicity


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(p: Path) -> dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_spectrum_h5(path: Path) -> dict[str, Any]:
    """Load Bloque B spectrum.h5 and extract M² eigenvalues."""
    import h5py

    with h5py.File(path, "r") as h5:
        data = {
            "M2": h5["M2"][:],                    # (n_delta, n_modes)
            "delta_uv": h5["delta_uv"][:],        # (n_delta,)
            "z_grid": h5["z_grid"][:],            # (n_z,)
            "d": int(h5.attrs.get("d", 3)),
            "L": float(h5.attrs.get("L", 1.0)),
            "n_modes": int(h5.attrs.get("n_modes", 10)),
            "bc_uv": str(h5.attrs.get("bc_uv", "dirichlet")),
            "bc_ir": str(h5.attrs.get("bc_ir", "dirichlet")),
        }
    return data


def _generate_open_bc_signal(
    omega_R: float,
    gamma: float,  # absorption parameter: ω_I = -gamma
    duration: float = 1.0,
    sample_rate: float = 4096.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate damped sinusoid with tunable absorption.

    h(t) = A * exp(-gamma * t) * cos(omega_R * t + phi)

    When gamma → 0: no decay (closed, Hermitian limit)
    When gamma > 0: exponential decay (open, horizon-like)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / sample_rate)
    phi = rng.uniform(0, 2 * np.pi)
    amplitude = 1.0

    if gamma > 0:
        signal = amplitude * np.exp(-gamma * t) * np.cos(omega_R * t + phi)
    else:
        # Pure oscillation (no decay)
        signal = amplitude * np.cos(omega_R * t + phi)

    # Add small noise
    noise_level = 0.001 * amplitude
    signal += rng.normal(0, noise_level, size=len(t))

    return t, signal


def _fit_damped_sinusoid(
    t: np.ndarray,
    signal: np.ndarray,
) -> dict[str, float]:
    """Fit damped sinusoid to extract omega_R and omega_I.

    Returns omega_R (frequency) and omega_I (decay rate, negative for decay).
    """
    dt = t[1] - t[0]
    n = len(signal)

    # FFT to estimate frequency
    freqs = np.fft.rfftfreq(n, dt)
    fft_mag = np.abs(np.fft.rfft(signal))
    peak_idx = np.argmax(fft_mag[1:]) + 1
    omega_R = 2 * np.pi * freqs[peak_idx]

    # Estimate decay from envelope
    analytic = np.abs(signal + 1j * np.imag(
        np.fft.ifft(np.fft.fft(signal) * (1 - np.sign(np.fft.fftfreq(n))))
    ))
    envelope = np.maximum(analytic, 1e-12)

    # Linear fit on log envelope
    log_env = np.log(envelope + 1e-12)
    n_fit = n // 2
    if n_fit < 10:
        n_fit = n
    coeffs = np.polyfit(t[:n_fit], log_env[:n_fit], 1)
    omega_I = coeffs[0]  # slope = omega_I (negative for decay)

    return {
        "omega_R": float(omega_R),
        "omega_I": float(omega_I),
        "omega_R_sq": float(omega_R ** 2),
    }


def _run_absorption_sweep(
    M2_target: float,
    gamma_values: list[float],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run QNM fit for different absorption levels.

    Uses M² from Bloque B to set omega_R = sqrt(M²), then varies gamma.
    """
    if M2_target <= 0:
        # Skip negative/zero eigenvalues
        return []

    omega_R_true = np.sqrt(M2_target)
    results = []

    for gamma in gamma_values:
        t, signal = _generate_open_bc_signal(
            omega_R=omega_R_true,
            gamma=gamma,
            duration=1.0,
            sample_rate=4096.0,
            seed=seed,
        )

        fit = _fit_damped_sinusoid(t, signal)

        results.append({
            "gamma": float(gamma),
            "omega_I_true": float(-gamma),
            "omega_R_true": float(omega_R_true),
            "M2_target": float(M2_target),
            "omega_R_fit": fit["omega_R"],
            "omega_I_fit": fit["omega_I"],
            "omega_R_sq_fit": fit["omega_R_sq"],
            "omega_R_sq_error": float(abs(fit["omega_R_sq"] - M2_target) / M2_target),
            "omega_I_error": float(abs(fit["omega_I"] - (-gamma))),
        })

    return results


def _evaluate_contract_c3(
    sweep_results: list[dict[str, Any]],
    omega_R_sq_rel_tol: float,
    omega_I_zero_tol: float,
) -> dict[str, Any]:
    """Contract C3: Closed limit recovery.

    PASS if:
      - When gamma ≈ 0: |ω_R² - M²| / M² < tol
      - When gamma ≈ 0: |ω_I| < omega_I_zero_tol
    """
    violations = []
    closed_cases = []

    for result in sweep_results:
        gamma = result["gamma"]

        # Check "closed" cases (small gamma)
        if gamma < 1.0:  # gamma < 1 is "mostly closed"
            closed_cases.append(result)

            omega_R_sq_error = result["omega_R_sq_error"]
            omega_I_fit = abs(result["omega_I_fit"])

            if omega_R_sq_error > omega_R_sq_rel_tol:
                violations.append({
                    "gamma": gamma,
                    "metric": "omega_R_sq_error",
                    "value": omega_R_sq_error,
                    "threshold": omega_R_sq_rel_tol,
                    "reason": "ω_R² does not match M² in closed limit",
                })

            if gamma < 0.1 and omega_I_fit > omega_I_zero_tol + gamma:
                # For very small gamma, omega_I should be close to -gamma
                violations.append({
                    "gamma": gamma,
                    "metric": "omega_I_fit",
                    "value": omega_I_fit,
                    "threshold": omega_I_zero_tol,
                    "reason": "ω_I does not approach 0 in closed limit",
                })

    return {
        "contract": "C3_closed_limit_recovery",
        "verdict": "PASS" if len(violations) == 0 else "FAIL",
        "thresholds": {
            "omega_R_sq_rel_tol": omega_R_sq_rel_tol,
            "omega_I_zero_tol": omega_I_zero_tol,
        },
        "n_closed_cases": len(closed_cases),
        "violations": violations,
    }


def _evaluate_contract_c4(
    sweep_results: list[dict[str, Any]],
    monotonicity_tol: float,
) -> dict[str, Any]:
    """Contract C4: Monotonicity of decay with absorption.

    PASS if:
      - |ω_I| increases (or stays constant) as gamma increases
      - i.e., |ω_I(gamma_i)| <= |ω_I(gamma_{i+1})| for all i
    """
    if len(sweep_results) < 2:
        return {
            "contract": "C4_monotonicity",
            "verdict": "PASS",
            "reason": "single point, monotonicity trivially satisfied",
            "violations": [],
        }

    # Sort by gamma
    sorted_results = sorted(sweep_results, key=lambda x: x["gamma"])

    violations = []
    for i in range(len(sorted_results) - 1):
        gamma_i = sorted_results[i]["gamma"]
        gamma_j = sorted_results[i + 1]["gamma"]
        omega_I_i = abs(sorted_results[i]["omega_I_fit"])
        omega_I_j = abs(sorted_results[i + 1]["omega_I_fit"])

        # |ω_I| should increase with gamma (more absorption → more decay)
        if omega_I_j < omega_I_i - monotonicity_tol:
            violations.append({
                "gamma_i": gamma_i,
                "gamma_j": gamma_j,
                "omega_I_i": omega_I_i,
                "omega_I_j": omega_I_j,
                "reason": "|ω_I| decreased when absorption increased",
            })

    return {
        "contract": "C4_monotonicity",
        "verdict": "PASS" if len(violations) == 0 else "FAIL",
        "thresholds": {
            "monotonicity_tol": monotonicity_tol,
        },
        "n_transitions": len(sorted_results) - 1,
        "violations": violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EXP_RINGDOWN_QNM_01 — Closed ↔ Open limit comparison"
    )
    ap.add_argument("--run", required=True, help="Run ID")
    ap.add_argument("--out-root", default="runs", help="Output root directory")
    ap.add_argument(
        "--in-spectrum",
        help="Path to Bloque B spectrum.h5 (default: auto-resolve from run)",
    )
    ap.add_argument(
        "--upstream-run",
        help="Upstream run ID containing spectrum.h5 (if different from --run)",
    )
    ap.add_argument(
        "--gamma-sweep",
        default="0.0,0.01,0.1,1.0,10.0,100.0,250.0",
        help="Absorption values to sweep (comma-separated)",
    )
    ap.add_argument(
        "--mode-indices",
        default="0,1,2",
        help="Which modes from M² to test (comma-separated indices)",
    )
    ap.add_argument(
        "--delta-index",
        type=int,
        default=0,
        help="Which delta slice to use from spectrum.h5",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Tolerance overrides
    ap.add_argument("--omega-r-sq-rel-tol", type=float, default=DEFAULT_OMEGA_R_SQ_REL_TOL)
    ap.add_argument("--omega-i-zero-tol", type=float, default=DEFAULT_OMEGA_I_ZERO_TOL)
    ap.add_argument("--monotonicity-tol", type=float, default=DEFAULT_MONOTONICITY_TOL)

    args = ap.parse_args()

    # Validate run
    try:
        out_root = resolve_out_root(args.out_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_dir = get_run_dir(args.run, base_dir=out_root)

    # Resolve spectrum path
    if args.in_spectrum:
        spectrum_path = Path(args.in_spectrum)
    elif args.upstream_run:
        upstream_dir = get_run_dir(args.upstream_run, base_dir=out_root)
        spectrum_path = resolve_spectrum_path(upstream_dir)
    else:
        # Try to find spectrum in current run
        try:
            spectrum_path = resolve_spectrum_path(run_dir)
        except FileNotFoundError:
            print("ERROR: No spectrum.h5 found. Provide --in-spectrum or --upstream-run", file=sys.stderr)
            return 2

    if not spectrum_path.exists():
        print(f"ERROR: spectrum.h5 not found at {spectrum_path}", file=sys.stderr)
        return 2

    # Create stage directory
    stage_name = "experiment/ringdown/EXP_RINGDOWN_QNM_01_closed_open_limit"
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, stage_name, base_dir=out_root)

    # Parse sweep parameters
    gamma_values = [float(x.strip()) for x in args.gamma_sweep.split(",")]
    mode_indices = [int(x.strip()) for x in args.mode_indices.split(",")]
    delta_idx = args.delta_index

    # Load Bloque B spectrum
    print(f"Loading spectrum from: {spectrum_path}")
    spectrum_data = _load_spectrum_h5(spectrum_path)
    spectrum_hash = sha256_file(spectrum_path)

    M2_matrix = spectrum_data["M2"]  # (n_delta, n_modes)
    n_delta, n_modes = M2_matrix.shape

    if delta_idx >= n_delta:
        print(f"ERROR: delta_index {delta_idx} out of range (n_delta={n_delta})", file=sys.stderr)
        return 2

    # Select M² values to test
    M2_slice = M2_matrix[delta_idx, :]
    delta_value = spectrum_data["delta_uv"][delta_idx]

    print(f"Using delta={delta_value:.3f}, modes={mode_indices}")
    print(f"Gamma sweep: {gamma_values}")

    # Run absorption sweep for each mode
    all_results = []
    per_mode_results = {}

    for mode_idx in mode_indices:
        if mode_idx >= n_modes:
            print(f"WARNING: mode_index {mode_idx} out of range, skipping", file=sys.stderr)
            continue

        M2_target = M2_slice[mode_idx]
        if M2_target <= 0:
            print(f"WARNING: M²[{mode_idx}] = {M2_target} <= 0, skipping", file=sys.stderr)
            continue

        print(f"  Mode {mode_idx}: M² = {M2_target:.6f}, ω_R = {np.sqrt(M2_target):.6f}")

        sweep = _run_absorption_sweep(
            M2_target=M2_target,
            gamma_values=gamma_values,
            seed=args.seed + mode_idx,
        )

        for r in sweep:
            r["mode_index"] = mode_idx
            r["delta"] = float(delta_value)

        all_results.extend(sweep)
        per_mode_results[f"mode_{mode_idx}"] = sweep

    if not all_results:
        print("ERROR: No valid modes to analyze", file=sys.stderr)
        return 2

    # Evaluate contracts
    c3_result = _evaluate_contract_c3(
        all_results,
        omega_R_sq_rel_tol=args.omega_r_sq_rel_tol,
        omega_I_zero_tol=args.omega_i_zero_tol,
    )
    c4_result = _evaluate_contract_c4(
        all_results,
        monotonicity_tol=args.monotonicity_tol,
    )

    # Overall verdict
    overall_verdict = "PASS" if (c3_result["verdict"] == "PASS" and c4_result["verdict"] == "PASS") else "FAIL"

    # Build comparison.json
    comparison = {
        "schema_version": "closed_open_comparison_v1",
        "created": utc_now_iso(),
        "spectrum_source": str(spectrum_path),
        "spectrum_sha256": spectrum_hash,
        "delta": float(delta_value),
        "delta_index": delta_idx,
        "mode_indices": mode_indices,
        "gamma_values": gamma_values,
        "n_results": len(all_results),
        "summary": {
            "closed_limit_valid": c3_result["verdict"] == "PASS",
            "monotonicity_valid": c4_result["verdict"] == "PASS",
        },
        "bloque_b": {
            "d": spectrum_data["d"],
            "L": spectrum_data["L"],
            "bc_uv": spectrum_data["bc_uv"],
            "bc_ir": spectrum_data["bc_ir"],
            "n_modes": spectrum_data["n_modes"],
        },
    }

    comparison_path = outputs_dir / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Write per-mode results
    per_mode_path = outputs_dir / "per_mode_results.json"
    with open(per_mode_path, "w", encoding="utf-8") as f:
        json.dump(per_mode_results, f, indent=2)

    # Build contract_verdict.json
    contract_verdict = {
        "schema_version": "contract_verdict_v1",
        "created": utc_now_iso(),
        "verdict": overall_verdict,
        "contracts": {
            "C3_closed_limit_recovery": c3_result,
            "C4_monotonicity": c4_result,
        },
        "inputs": {
            "spectrum": {
                "path": str(spectrum_path),
                "sha256": spectrum_hash,
            },
        },
        "interpretation": {
            "C3": "When absorption → 0, open-BC model recovers closed-BC spectrum (ω_R² → M²)",
            "C4": "Decay rate |ω_I| increases monotonically with absorption parameter",
        },
    }

    contract_verdict_path = outputs_dir / "contract_verdict.json"
    with open(contract_verdict_path, "w", encoding="utf-8") as f:
        json.dump(contract_verdict, f, indent=2)

    # Write stage_summary
    summary = {
        "stage": stage_name,
        "script": "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
        "params": {
            "gamma_sweep": args.gamma_sweep,
            "mode_indices": args.mode_indices,
            "delta_index": delta_idx,
            "seed": args.seed,
            "omega_r_sq_rel_tol": args.omega_r_sq_rel_tol,
            "omega_i_zero_tol": args.omega_i_zero_tol,
            "monotonicity_tol": args.monotonicity_tol,
        },
        "inputs": {
            "spectrum": {
                "path": str(spectrum_path),
                "sha256": spectrum_hash,
            },
        },
        "outputs": {
            "comparison": "outputs/comparison.json",
            "per_mode_results": "outputs/per_mode_results.json",
            "contract_verdict": "outputs/contract_verdict.json",
        },
        "verdict": overall_verdict,
    }
    write_stage_summary(stage_dir, summary)

    # Write manifest
    artifacts = {
        "comparison": comparison_path,
        "per_mode_results": per_mode_path,
        "contract_verdict": contract_verdict_path,
    }
    write_manifest(stage_dir, artifacts, extra={"verdict": overall_verdict})

    # Print summary
    print(f"\nStage completed: {stage_name}")
    print(f"  Verdict: {overall_verdict}")
    print(f"  C3 (closed limit): {c3_result['verdict']}")
    print(f"  C4 (monotonicity): {c4_result['verdict']}")

    if c3_result["violations"]:
        print(f"  C3 violations: {len(c3_result['violations'])}")
    if c4_result["violations"]:
        print(f"  C4 violations: {len(c4_result['violations'])}")

    if overall_verdict != "PASS":
        return EXIT_CONTRACT_FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
