#!/usr/bin/env python3
"""
Stage BASURIN: Holographic Spectral Classifier (HSC) Detector.

Contrato: runs/<run_id>/experiment/hsc_detector/

Este stage analiza datos de CFT (Conformal Field Theory) para detectar
si el espectro y coeficientes OPE son consistentes con una teoría holográfica
con dual de gravedad local en el bulk (Local Bulk Candidate).

Fases:
  - Fase 1: Análisis espectral (scalar gap, densidad de operadores)
  - Fase 2: Análisis de coeficientes OPE (jerarquía λ_OOO vs λ_OO[OO]n)

Inputs:
  - runs/<run_id>/inputs/hsc/input.json (esquema hsc_input_v1)
  - runs/<run_id>/inputs/hsc/thresholds.json (opcional)

Outputs:
  - manifest.json
  - stage_summary.json
  - outputs/verdict.json
  - outputs/phase_1_spectrum_analysis.json
  - outputs/phase_2_ope_analysis.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.1.0"

# ----------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_UNDERDETERMINED = "UNDERDETERMINED"

OVERALL_PASS_LOCAL_BULK_CANDIDATE = "PASS_LOCAL_BULK_CANDIDATE"
OVERALL_FAIL_LOCAL_BULK = "FAIL_LOCAL_BULK"
OVERALL_UNDERDETERMINED = "UNDERDETERMINED"


# ----------------------------------------------------------------------
# Phase 1: Spectrum Analysis
# ----------------------------------------------------------------------
def phase_1_verdict(
    spectrum: dict[str, Any],
    d: float,
    window_max: float = 5.0,
    max_density: int = 5,
) -> tuple[str, dict[str, Any]]:
    """
    Analiza el espectro de operadores escalares.

    Criterios:
      - PASS: low_lying_count <= max_density
      - FAIL: low_lying_count > max_density * 2
      - UNDERDETERMINED: otherwise

    Args:
        spectrum: dict con key "operators" (lista de operadores)
        d: dimensión espaciotemporal
        window_max: límite superior de la ventana de análisis
        max_density: máximo número de operadores permitidos en ventana para PASS

    Returns:
        (verdict, features) donde features contiene métricas calculadas
    """
    operators = spectrum.get("operators", [])

    # Filtrar operadores escalares (spin=0) con dimensión > 0
    scalars = sorted([
        op["dim"]
        for op in operators
        if op.get("spin", 0) == 0 and op.get("dim", 0) > 1e-9
    ])

    if not scalars:
        return VERDICT_UNDERDETERMINED, {
            "reason": "No scalar operators above identity",
            "scalar_gap": None,
            "low_lying_density": 0,
            "sparsity_ratio": None,
        }

    scalar_gap = scalars[0]
    low_lying_count = sum(1 for dim in scalars if dim < window_max)

    # Evita división por cero: ventana efectiva mínima de 1 unidad
    effective_window = max(1e-9, window_max - d / 2)
    sparsity_ratio = low_lying_count / effective_window

    if low_lying_count <= max_density:
        verdict = VERDICT_PASS
    elif low_lying_count > max_density * 2:
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_UNDERDETERMINED

    return verdict, {
        "scalar_gap": scalar_gap,
        "low_lying_density": low_lying_count,
        "sparsity_ratio": round(sparsity_ratio, 6),
    }


# ----------------------------------------------------------------------
# Phase 2: OPE Coefficient Analysis
# ----------------------------------------------------------------------
def phase_2_verdict(
    ope_coeffs: dict[str, float],
    conventions: dict[str, Any],
    min_hierarchy: float = 3.0,
) -> tuple[str, dict[str, Any]]:
    """
    Analiza la jerarquía de coeficientes OPE.

    Clasificación basada en conventions:
      - light_ops: operadores ligeros (e.g., ["sigma", "epsilon"])
      - tower_ops_prefix: prefijos de operadores de torre (e.g., ["[sigma sigma]_"])

    λ_OOO: coeficientes donde los 3 operadores son light_ops
    λ_OO[OO]n: coeficientes donde op1, op2 son light y op3 es tower

    Criterios:
      - PASS: hierarchy_ratio >= min_hierarchy
      - FAIL: hierarchy_ratio <= 1.0
      - UNDERDETERMINED: otherwise

    Args:
        ope_coeffs: dict {key: value} donde key = "op1_op2_op3"
        conventions: dict con light_ops y tower_ops_prefix
        min_hierarchy: ratio mínimo λ_OOO/λ_tower para PASS

    Returns:
        (verdict, features) donde features contiene métricas calculadas
    """
    light_ops = set(conventions.get("light_ops", []))
    tower_prefixes = conventions.get("tower_ops_prefix", [])

    if not light_ops:
        return VERDICT_UNDERDETERMINED, {
            "reason": "No light_ops defined in conventions",
            "median_lambda_OOO": None,
            "median_lambda_OO_tower": None,
            "hierarchy_ratio_R": None,
        }

    lambda_OOO: list[float] = []
    lambda_tower: list[float] = []

    for key, value in ope_coeffs.items():
        # Parse key format: "op1_op2_op3"
        # Handle composite operator names with brackets
        parts = _parse_ope_key(key)
        if len(parts) != 3:
            continue

        op1, op2, op3 = parts

        # Clasificación determinista usando conventions
        ops_set = {op1, op2, op3}
        if ops_set.issubset(light_ops):
            lambda_OOO.append(abs(value))
        elif (
            any(op3.startswith(prefix) for prefix in tower_prefixes)
            and {op1, op2}.issubset(light_ops)
        ):
            lambda_tower.append(abs(value))

    if not lambda_OOO or not lambda_tower:
        return VERDICT_UNDERDETERMINED, {
            "reason": "Missing OPE classes per conventions",
            "median_lambda_OOO": _safe_median(lambda_OOO),
            "median_lambda_OO_tower": _safe_median(lambda_tower),
            "hierarchy_ratio_R": None,
            "lambda_OOO_count": len(lambda_OOO),
            "lambda_tower_count": len(lambda_tower),
        }

    median_OOO = _median(lambda_OOO)
    median_tower = _median(lambda_tower)

    if median_tower == 0:
        hierarchy_ratio = float("inf")
    else:
        hierarchy_ratio = median_OOO / median_tower

    if hierarchy_ratio >= min_hierarchy:
        verdict = VERDICT_PASS
    elif hierarchy_ratio <= 1.0:
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_UNDERDETERMINED

    return verdict, {
        "median_lambda_OOO": round(median_OOO, 6),
        "median_lambda_OO_tower": round(median_tower, 6),
        "hierarchy_ratio_R": (
            round(hierarchy_ratio, 4) if hierarchy_ratio != float("inf") else "inf"
        ),
        "lambda_OOO_count": len(lambda_OOO),
        "lambda_tower_count": len(lambda_tower),
    }


def _parse_ope_key(key: str) -> list[str]:
    """
    Parse OPE key format "op1_op2_op3" handling composite names with brackets.

    Handles tower operator suffixes like "[sigma sigma]_0" as a single operator.

    Examples:
        "sigma_sigma_sigma" -> ["sigma", "sigma", "sigma"]
        "sigma_sigma_[sigma sigma]_0" -> ["sigma", "sigma", "[sigma sigma]_0"]
    """
    parts: list[str] = []
    current = ""
    bracket_depth = 0
    just_closed_bracket = False

    i = 0
    while i < len(key):
        char = key[i]

        if char == "[":
            bracket_depth += 1
            current += char
            just_closed_bracket = False
        elif char == "]":
            bracket_depth -= 1
            current += char
            just_closed_bracket = True
        elif char == "_" and bracket_depth == 0:
            # Check if this underscore is part of a tower suffix (e.g., "_0", "_1")
            # after a bracket close
            if just_closed_bracket and i + 1 < len(key) and key[i + 1].isdigit():
                # This is a tower index suffix, include it in current operator
                current += char
                just_closed_bracket = False
            else:
                if current:
                    parts.append(current)
                    current = ""
                just_closed_bracket = False
        else:
            current += char
            if not char.isdigit():
                just_closed_bracket = False

        i += 1

    if current:
        parts.append(current)

    return parts


def _median(values: list[float]) -> float:
    """Calculate median of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def _safe_median(values: list[float]) -> Optional[float]:
    """Calculate median or return None if empty."""
    if not values:
        return None
    return round(_median(values), 6)


# ----------------------------------------------------------------------
# Overall Verdict
# ----------------------------------------------------------------------
def overall_verdict(
    phase_1_result: tuple[str, dict[str, Any]],
    phase_2_result: tuple[str, dict[str, Any]],
) -> str:
    """
    Combina veredictos de ambas fases.

    - PASS_LOCAL_BULK_CANDIDATE: ambas fases PASS
    - FAIL_LOCAL_BULK: cualquier fase FAIL
    - UNDERDETERMINED: ninguna FAIL, al menos una UNDERDETERMINED
    """
    p1_verdict, _ = phase_1_result
    p2_verdict, _ = phase_2_result

    if p1_verdict == VERDICT_FAIL or p2_verdict == VERDICT_FAIL:
        return OVERALL_FAIL_LOCAL_BULK
    if p1_verdict == VERDICT_PASS and p2_verdict == VERDICT_PASS:
        return OVERALL_PASS_LOCAL_BULK_CANDIDATE
    return OVERALL_UNDERDETERMINED


# ----------------------------------------------------------------------
# File I/O
# ----------------------------------------------------------------------
def load_input(input_path: Path) -> tuple[dict[str, Any], str]:
    """Load input JSON and return (data, sha256_hash)."""
    with open(input_path, "rb") as f:
        input_bytes = f.read()
    input_hash = "sha256:" + hashlib.sha256(input_bytes).hexdigest()
    input_data = json.loads(input_bytes.decode("utf-8"))
    return input_data, input_hash


def load_thresholds(
    thresholds_path: Optional[Path],
) -> dict[str, dict[str, Any]]:
    """Load thresholds JSON or return defaults."""
    defaults: dict[str, dict[str, Any]] = {
        "phase_1": {"window_max": 5.0, "max_density": 5},
        "phase_2": {"min_hierarchy": 3.0},
    }
    if thresholds_path and thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            custom = json.load(f)
        # Merge custom into defaults
        for phase_key in ["phase_1", "phase_2"]:
            if phase_key in custom:
                defaults[phase_key].update(custom[phase_key])
    return defaults


def get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
        return "git:" + result.stdout.strip()
    except Exception:
        return None


# ----------------------------------------------------------------------
# Stage Entry Point
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HSC Detector: Holographic Spectral Classifier stage"
    )
    parser.add_argument(
        "--run",
        required=True,
        help="run_id under runs/<run_id>",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input.json (hsc_input_v1 schema)",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional path to thresholds.json",
    )
    parser.add_argument(
        "--out-root",
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for HSC Detector stage."""
    args = parse_args()

    # Check BASURIN abort condition
    run_valid = os.getenv("RUN_VALID", "PASS")
    if run_valid != "PASS":
        print(f"[BASURIN ABORT] RUN_VALID={run_valid}", file=sys.stderr)
        return 1

    # Validate paths
    try:
        out_root = resolve_out_root(args.out_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    thresholds_path = Path(args.thresholds) if args.thresholds else None

    # Create stage directories
    stage_dir, outputs_dir = ensure_stage_dirs(
        args.run, "hsc_detector", base_dir=out_root
    )
    run_dir = stage_dir.parent

    # Load input data
    input_data, input_hash = load_input(input_path)

    # Validate input schema
    if "metadata" not in input_data:
        print("ERROR: Input missing 'metadata' field", file=sys.stderr)
        return 2
    if "spectrum" not in input_data:
        print("ERROR: Input missing 'spectrum' field", file=sys.stderr)
        return 2
    if "ope_coefficients" not in input_data:
        print("ERROR: Input missing 'ope_coefficients' field", file=sys.stderr)
        return 2

    metadata = input_data["metadata"]
    d = metadata.get("d", 3)  # Default to d=3
    conventions = metadata.get("conventions", {})

    # Load thresholds
    thresholds = load_thresholds(thresholds_path)

    # Execute phases
    p1_verdict, p1_features = phase_1_verdict(
        input_data["spectrum"],
        d,
        **thresholds["phase_1"],
    )
    p2_verdict, p2_features = phase_2_verdict(
        input_data["ope_coefficients"],
        conventions,
        **thresholds["phase_2"],
    )

    # Compute overall verdict
    overall = overall_verdict((p1_verdict, p1_features), (p2_verdict, p2_features))

    # Build warnings
    warnings: list[str] = []
    if p1_verdict == VERDICT_UNDERDETERMINED:
        reason = p1_features.get("reason", "")
        if reason:
            warnings.append(f"Phase 1: {reason}")
    if p2_verdict == VERDICT_UNDERDETERMINED:
        reason = p2_features.get("reason", "")
        if reason:
            warnings.append(f"Phase 2: {reason}")

    # Determine thresholds source
    thresholds_source = (
        str(thresholds_path.relative_to(run_dir))
        if thresholds_path and thresholds_path.exists()
        else "internal_defaults"
    )

    # Compute relative input path
    try:
        input_rel = str(input_path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        input_rel = str(input_path)

    # Build output artifacts
    timestamp = utc_now_iso()
    git_commit = get_git_commit()

    # --- verdict.json ---
    verdict_data = {
        "run_id": args.run,
        "input_path": input_rel,
        "input_data_hash": input_hash,
        "overall_verdict": overall,
        "phase_summary": {
            "phase_1": {
                "verdict": p1_verdict,
                "features": p1_features,
                "thresholds_applied": thresholds["phase_1"],
            },
            "phase_2": {
                "verdict": p2_verdict,
                "features": p2_features,
                "thresholds_applied": thresholds["phase_2"],
            },
        },
        "warnings": warnings,
        "provenance": {
            "stage": "experiment/hsc_detector",
            "thresholds_source": thresholds_source,
            "code_commit": git_commit,
            "version": __version__,
        },
    }
    verdict_path = outputs_dir / "verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, indent=2)

    # --- phase_1_spectrum_analysis.json ---
    phase_1_data = {
        "phase": "phase_1_spectrum_analysis",
        "run_id": args.run,
        "timestamp": timestamp,
        "input_spectrum": {
            "operator_count": len(input_data["spectrum"].get("operators", [])),
            "d": d,
        },
        "verdict": p1_verdict,
        "features": p1_features,
        "thresholds": thresholds["phase_1"],
    }
    phase_1_path = outputs_dir / "phase_1_spectrum_analysis.json"
    with open(phase_1_path, "w", encoding="utf-8") as f:
        json.dump(phase_1_data, f, indent=2)

    # --- phase_2_ope_analysis.json ---
    phase_2_data = {
        "phase": "phase_2_ope_analysis",
        "run_id": args.run,
        "timestamp": timestamp,
        "input_ope": {
            "coefficient_count": len(input_data["ope_coefficients"]),
            "conventions": conventions,
        },
        "verdict": p2_verdict,
        "features": p2_features,
        "thresholds": thresholds["phase_2"],
    }
    phase_2_path = outputs_dir / "phase_2_ope_analysis.json"
    with open(phase_2_path, "w", encoding="utf-8") as f:
        json.dump(phase_2_data, f, indent=2)

    # --- manifest.json ---
    artifacts = {
        "verdict": verdict_path,
        "phase_1_analysis": phase_1_path,
        "phase_2_analysis": phase_2_path,
    }
    write_manifest(
        stage_dir,
        artifacts,
        extra={
            "version": __version__,
            "schema": "hsc_detector_v1",
        },
    )

    # --- stage_summary.json ---
    summary = {
        "stage_name": "hsc_detector",
        "run_id": args.run,
        "timestamp": timestamp,
        "version": __version__,
        "inputs": [
            {
                "path": input_rel,
                "sha256": input_hash.replace("sha256:", ""),
                "role": "primary",
            }
        ],
        "parameters": thresholds,
        "results": {
            "overall_verdict": overall,
            "phase_verdicts": {
                "phase_1": p1_verdict,
                "phase_2": p2_verdict,
            },
            "outputs": {
                "verdict": "outputs/verdict.json",
                "phase_1_analysis": "outputs/phase_1_spectrum_analysis.json",
                "phase_2_analysis": "outputs/phase_2_ope_analysis.json",
            },
        },
    }
    write_stage_summary(stage_dir, summary)

    # Success message
    print(f"[HSC DETECTOR] {overall}")
    print(f"  Phase 1: {p1_verdict}")
    print(f"  Phase 2: {p2_verdict}")
    print(f"  Output: {stage_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
