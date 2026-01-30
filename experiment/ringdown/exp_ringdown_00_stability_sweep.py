#!/usr/bin/env python3
"""
BASURIN — EXP_RINGDOWN_00: Stability Sweep

Gate experiment that validates ringdown parameter extraction is robust
against variations in preprocessing (window, bandpass, whitening).

Schema: contracts/EXP_RINGDOWN_00_SCHEMA.md

Inputs:
    - runs/<run_id>/RUN_VALID/outputs/run_valid.json (verdict=PASS)
    - runs/<run_id>/ringdown_synth/outputs/synthetic_event.json

Outputs:
    runs/<run_id>/experiment/exp_ringdown_00_stability/
        ├── manifest.json
        ├── stage_summary.json
        └── outputs/
            ├── sweep_plan.json
            ├── diagnostics.json
            ├── contract_verdict.json
            └── per_case/
                └── case_XXX.json

Usage:
    python experiment/ringdown/exp_ringdown_00_stability_sweep.py \
        --run <run_id> --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import (
    get_runs_root,
    require_run_valid,
    sha256_file,
    utc_now_iso,
    write_manifest,
    write_stage_summary,
)

# =============================================================================
# Constants from schema
# =============================================================================

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_NAME = "exp_ringdown_00_stability"
STAGE_NAME = "exp_ringdown_00_stability"

# Tolerances (from schema C2)
REL_TOL_F220 = 0.02  # 2%
REL_TOL_TAU220 = 0.05  # 5%
Q_CONSISTENCY_TOL = 0.02  # 2% for Q consistency check

# SNR policy (from schema C3)
SNR_MIN = 8.0
SNR_POLICY = "SKIP_LOW_SNR"

# Pass rule (from schema C4)
PASS_RULE = "ALL"

# Seed (from schema F1)
SEED_GLOBAL = 42


@dataclass(frozen=True)
class CaseParams:
    """Parameters for a single sweep case."""

    case_id: str
    t_ref_shift: float  # seconds, relative offset
    duration: float  # seconds
    f_low: int  # Hz
    f_high: int  # Hz
    whitening_method: str


def build_sweep_plan(seed_global: int) -> list[CaseParams]:
    """Build the OFAT sweep grid (8 cases).

    Schema D2: One Factor At a Time from baseline.
    """
    baseline = CaseParams(
        case_id="case_000",
        t_ref_shift=0.0,
        duration=0.5,
        f_low=20,
        f_high=500,
        whitening_method="median_psd",
    )

    cases = [
        baseline,
        CaseParams("case_001", -0.05, 0.5, 20, 500, "median_psd"),  # shift -
        CaseParams("case_002", +0.05, 0.5, 20, 500, "median_psd"),  # shift +
        CaseParams("case_003", 0.0, 0.3, 20, 500, "median_psd"),  # duration short
        CaseParams("case_004", 0.0, 0.8, 20, 500, "median_psd"),  # duration long
        CaseParams("case_005", 0.0, 0.5, 30, 500, "median_psd"),  # f_low higher
        CaseParams("case_006", 0.0, 0.5, 20, 400, "median_psd"),  # f_high lower
        CaseParams("case_007", 0.0, 0.5, 20, 500, "welch_psd"),  # whitening alt
    ]

    return cases


def simulate_ringdown_analysis(
    synth_event: dict,
    params: CaseParams,
    seed: int,
) -> dict[str, Any]:
    """Simulate ringdown parameter extraction for a case.

    In a real implementation, this would call the actual Bayesian inference.
    Here we simulate with controlled noise to test the contract infrastructure.

    Returns:
        Dictionary with extracted metrics and status.
    """
    np.random.seed(seed)

    truth = synth_event["qnm_truth"]
    signal_props = synth_event["signal_properties"]

    f_truth = truth["f_220_hz"]
    tau_truth = truth["tau_220_ms"]
    Q_truth = truth["Q_220"]
    snr_nominal = signal_props["snr_nominal"]

    # Check if window is valid (no padding required)
    t_ref = signal_props["t_ref"]
    duration_available = signal_props["duration_available"]
    window_start = t_ref + params.t_ref_shift
    window_end = window_start + params.duration

    if window_start < 0 or window_end > duration_available:
        return {
            "case_id": params.case_id,
            "status": "SKIP_NO_DATA",
            "reason": f"Window [{window_start:.3f}, {window_end:.3f}] exceeds available data [0, {duration_available}]",
            "snr_effective": None,
            "metrics": None,
            "deviations_from_baseline": None,
            "violations": [],
        }

    # Simulate SNR degradation based on parameters
    snr_factor = 1.0
    if params.duration < 0.5:
        snr_factor *= params.duration / 0.5
    if params.f_high < 500:
        snr_factor *= 0.95
    if params.f_low > 20:
        snr_factor *= 0.98

    snr_effective = snr_nominal * snr_factor

    # Check SNR threshold
    if snr_effective < SNR_MIN:
        return {
            "case_id": params.case_id,
            "status": "SKIP_LOW_SNR",
            "reason": f"snr_effective={snr_effective:.2f} < snr_min={SNR_MIN}",
            "snr_effective": snr_effective,
            "metrics": None,
            "deviations_from_baseline": None,
            "violations": [],
        }

    # Simulate extracted parameters with realistic scatter
    # Scatter scales inversely with SNR
    scatter_f = 0.005 * (25.0 / snr_effective)  # ~0.5% at SNR=25
    scatter_tau = 0.015 * (25.0 / snr_effective)  # ~1.5% at SNR=25

    # Add small systematic shifts based on preprocessing
    bias_f = 0.0
    bias_tau = 0.0

    if params.whitening_method == "welch_psd":
        bias_f = 0.003  # Small systematic shift
    if params.f_low > 20:
        bias_f += 0.002
    if params.duration < 0.5:
        bias_tau += 0.01

    f_extracted = f_truth * (1.0 + bias_f + np.random.normal(0, scatter_f))
    tau_extracted = tau_truth * (1.0 + bias_tau + np.random.normal(0, scatter_tau))

    # Compute Q and check consistency
    Q_computed = np.pi * f_extracted * (tau_extracted / 1000.0)  # tau is in ms
    Q_derived = np.pi * f_truth * (tau_truth / 1000.0)

    # Generate CI68 (mock)
    ci68_f = [f_extracted * 0.99, f_extracted * 1.01]
    ci68_tau = [tau_extracted * 0.97, tau_extracted * 1.03]

    q_consistency = "OK"
    if abs(Q_computed - Q_derived) / Q_derived > Q_CONSISTENCY_TOL:
        q_consistency = "WARN"

    metrics = {
        "f_220": {"median": f_extracted, "ci68": ci68_f},
        "tau_220": {"median": tau_extracted, "ci68": ci68_tau},
        "Q_220": {
            "computed": Q_computed,
            "derived": Q_derived,
            "consistency": q_consistency,
        },
    }

    return {
        "case_id": params.case_id,
        "status": "OK",
        "snr_effective": snr_effective,
        "metrics": metrics,
        "deviations_from_baseline": None,  # Filled in later
        "violations": [],  # Filled in later
    }


def compute_deviations_and_check(
    result: dict[str, Any],
    baseline_result: dict[str, Any],
) -> dict[str, Any]:
    """Compute deviations from baseline and check tolerances."""
    if result["status"] != "OK" or baseline_result["status"] != "OK":
        return result

    f_baseline = baseline_result["metrics"]["f_220"]["median"]
    tau_baseline = baseline_result["metrics"]["tau_220"]["median"]

    f_current = result["metrics"]["f_220"]["median"]
    tau_current = result["metrics"]["tau_220"]["median"]

    f_rel = abs(f_current - f_baseline) / f_baseline
    tau_rel = abs(tau_current - tau_baseline) / tau_baseline

    result["deviations_from_baseline"] = {
        "f_220_rel": f_rel,
        "tau_220_rel": tau_rel,
    }

    violations = []
    if f_rel > REL_TOL_F220:
        violations.append(
            {
                "metric": "f_220",
                "rel_deviation": f_rel,
                "rel_tol": REL_TOL_F220,
                "message": f"f_220 deviation {f_rel:.4f} > {REL_TOL_F220}",
            }
        )
    if tau_rel > REL_TOL_TAU220:
        violations.append(
            {
                "metric": "tau_220",
                "rel_deviation": tau_rel,
                "rel_tol": REL_TOL_TAU220,
                "message": f"tau_220 deviation {tau_rel:.4f} > {REL_TOL_TAU220}",
            }
        )

    result["violations"] = violations
    return result


def build_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build diagnostics summary from all case results."""
    valid_results = [r for r in results if r["status"] == "OK"]
    skipped_low_snr = [r for r in results if r["status"] == "SKIP_LOW_SNR"]
    skipped_no_data = [r for r in results if r["status"] == "SKIP_NO_DATA"]

    max_f_dev = 0.0
    max_tau_dev = 0.0

    for r in valid_results:
        if r["deviations_from_baseline"]:
            max_f_dev = max(max_f_dev, r["deviations_from_baseline"]["f_220_rel"])
            max_tau_dev = max(max_tau_dev, r["deviations_from_baseline"]["tau_220_rel"])

    return {
        "schema_version": SCHEMA_VERSION,
        "total_cases": len(results),
        "valid_cases": len(valid_results),
        "skipped_low_snr": len(skipped_low_snr),
        "skipped_no_data": len(skipped_no_data),
        "max_deviations": {
            "f_220_rel": max_f_dev,
            "tau_220_rel": max_tau_dev,
        },
        "tolerances": {
            "f_220_rel_tol": REL_TOL_F220,
            "tau_220_rel_tol": REL_TOL_TAU220,
        },
        "skipped_cases": [r["case_id"] for r in skipped_low_snr + skipped_no_data],
    }


def build_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build contract verdict from all case results."""
    valid_results = [r for r in results if r["status"] == "OK"]
    skipped = [r for r in results if r["status"] in ("SKIP_LOW_SNR", "SKIP_NO_DATA")]

    all_violations = []
    for r in valid_results:
        if r["violations"]:
            all_violations.extend(
                [{"case_id": r["case_id"], **v} for v in r["violations"]]
            )

    verdict = "PASS" if len(all_violations) == 0 else "FAIL"

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "verdict": verdict,
        "summary": {
            "total_cases": len(results),
            "valid_cases": len(valid_results),
            "skipped_cases": len(skipped),
            "violations": len(all_violations),
        },
        "assumptions": [
            f"snr_min={SNR_MIN} with {SNR_POLICY} policy",
            "t_ref_shift interpreted as relative offset (no silent padding)",
            f"Tolerances: f_220={REL_TOL_F220*100}%, tau_220={REL_TOL_TAU220*100}%",
        ],
        "violations_detail": all_violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP_RINGDOWN_00: Stability Sweep Gate"
    )
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument("--seed", type=int, default=SEED_GLOBAL, help="Random seed")
    parser.add_argument("--runs-root", default=None, help="Runs root directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runs_root = Path(args.runs_root) if args.runs_root else get_runs_root()
    run_dir = runs_root / args.run

    # =========================================================================
    # Gate 1: Require RUN_VALID == PASS
    # =========================================================================
    try:
        require_run_valid(runs_root, args.run)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    # =========================================================================
    # Gate 2: Require synthetic event
    # =========================================================================
    synth_path = run_dir / "ringdown_synth" / "outputs" / "synthetic_event.json"
    if not synth_path.exists():
        print(
            f"[BASURIN ABORT] Missing required input: {synth_path}",
            file=sys.stderr,
        )
        return 2

    with open(synth_path, "r", encoding="utf-8") as f:
        synth_event = json.load(f)

    # =========================================================================
    # Create output directories
    # =========================================================================
    stage_dir = run_dir / "experiment" / STAGE_NAME
    outputs_dir = stage_dir / "outputs"
    per_case_dir = outputs_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Build sweep plan
    # =========================================================================
    cases = build_sweep_plan(args.seed)

    sweep_plan = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "n_cases": len(cases),
        "baseline_case": "case_000",
        "seed_global": args.seed,
        "cases": [
            {
                "case_id": c.case_id,
                "seed": args.seed + i,
                "params": asdict(c),
            }
            for i, c in enumerate(cases)
        ],
    }

    sweep_plan_path = outputs_dir / "sweep_plan.json"
    with open(sweep_plan_path, "w", encoding="utf-8") as f:
        json.dump(sweep_plan, f, indent=2, sort_keys=True)
        f.write("\n")

    # =========================================================================
    # Execute sweep
    # =========================================================================
    results: list[dict[str, Any]] = []

    for i, case_params in enumerate(cases):
        seed_case = args.seed + i
        result = simulate_ringdown_analysis(synth_event, case_params, seed_case)
        results.append(result)

    # Compute deviations from baseline
    baseline_result = results[0]
    for result in results:
        compute_deviations_and_check(result, baseline_result)

    # Write per-case results
    for result in results:
        case_path = per_case_dir / f"{result['case_id']}.json"
        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")

    # =========================================================================
    # Build diagnostics and verdict
    # =========================================================================
    diagnostics = build_diagnostics(results)
    diagnostics_path = outputs_dir / "diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
        f.write("\n")

    verdict_data = build_verdict(results)
    verdict_path = outputs_dir / "contract_verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, indent=2, sort_keys=True)
        f.write("\n")

    # =========================================================================
    # Write stage_summary and manifest
    # =========================================================================
    summary = {
        "stage": STAGE_NAME,
        "run": args.run,
        "version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "seed": args.seed,
        "inputs": {
            "synthetic_event": str(synth_path.relative_to(runs_root)),
            "synthetic_event_sha256": sha256_file(synth_path),
        },
        "outputs": {
            "sweep_plan": "outputs/sweep_plan.json",
            "diagnostics": "outputs/diagnostics.json",
            "contract_verdict": "outputs/contract_verdict.json",
            "per_case": "outputs/per_case/",
        },
        "results": {
            "verdict": verdict_data["verdict"],
            "total_cases": verdict_data["summary"]["total_cases"],
            "valid_cases": verdict_data["summary"]["valid_cases"],
            "violations": verdict_data["summary"]["violations"],
        },
        "verdict": verdict_data["verdict"],
    }
    write_stage_summary(stage_dir, summary)

    # Collect all artifacts for manifest
    artifacts = {
        "sweep_plan": sweep_plan_path,
        "diagnostics": diagnostics_path,
        "contract_verdict": verdict_path,
    }
    for result in results:
        case_path = per_case_dir / f"{result['case_id']}.json"
        artifacts[f"per_case_{result['case_id']}"] = case_path

    write_manifest(stage_dir, artifacts, extra={"version": SCHEMA_VERSION})

    # =========================================================================
    # Console output
    # =========================================================================
    print(f"[{EXPERIMENT_NAME}] {verdict_data['verdict']}")
    print(f"  run: {args.run}")
    print(f"  cases: {len(results)} total, {diagnostics['valid_cases']} valid")
    print(f"  skipped: {diagnostics['skipped_low_snr']} low_snr, {diagnostics['skipped_no_data']} no_data")
    print(f"  max_dev: f={diagnostics['max_deviations']['f_220_rel']:.4f}, tau={diagnostics['max_deviations']['tau_220_rel']:.4f}")
    print(f"  violations: {verdict_data['summary']['violations']}")
    print(f"  output: {stage_dir}")

    return 0 if verdict_data["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
