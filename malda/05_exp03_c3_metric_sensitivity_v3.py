#!/usr/bin/env python3
"""
BASURIN — Experimento 03: Sensibilidad de C3 a la métrica
=========================================================

Objetivo
--------
Demostrar que el contrato C3 es sensible a la elección de norma y ponderación,
de modo que el mismo dataset mixto (ir_mix) puede:
  - FAIL con configuración naive (control negativo)
  - PASS con configuración robusta (control positivo)

Comandos canónicos (según documento Exp 03)
-------------------------------------------
NEGATIVO (A):
    python 04_diccionario.py --run ir_mix__A \\
      --enable-c3 --k-features 4 --n-bootstrap 0 \\
      --c3-metric rmse --c3-weights none --c3-threshold 0.05

POSITIVO (B):
    python 04_diccionario.py --run ir_mix__B \\
      --enable-c3 --k-features 4 --n-bootstrap 0 \\
      --c3-metric rmse_log --c3-weights inv_n4 \\
      --c3-adaptive-threshold --c3-threshold 0.02

Estructura de outputs
---------------------
runs/ir_mix/exp03/                    (o exp03_<label> si --variant-label)
├── manifest.json                     (índice de artefactos + hashes)
├── stage_summary.json                (params, timestamps, git_commit, hipótesis)
└── outputs/
    ├── metrics.json                  (métricas extraídas + configs + hypothesis_check)
    ├── validation_A.json             (copia de evidencia)
    ├── validation_B.json             (copia de evidencia)
    ├── dict_stage_summary_A.json     (copia de metadatos)
    ├── dict_stage_summary_B.json     (copia de metadatos)
    ├── log_A.txt                     (stdout+stderr de corrida A)
    └── log_B.txt                     (stdout+stderr de corrida B)

Uso
---
# Ejecución canónica (defaults del documento):
python 05_exp03_c3_metric_sensitivity.py

# Ejecución exploratoria (no sobrescribe stage canónico):
python 05_exp03_c3_metric_sensitivity.py --variant-label exploratory_thr04 --pos-threshold 0.04

Dependencias: h5py, numpy (ya usados en el repo)
Compatibilidad: Python 3.10+
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__version__ = "0.3.0"

# =============================================================================
# CANONICAL PARAMETERS (from Exp 03 document)
# =============================================================================
CANONICAL_PROFILES = "vacuum,crust,mantle,core,mixed"
CANONICAL_NEG_THRESHOLD = 0.05
CANONICAL_POS_THRESHOLD = 0.02
CANONICAL_K_FEATURES = 4
CANONICAL_N_BOOTSTRAP = 0


# =============================================================================
# Utility functions
# =============================================================================

def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> Optional[str]:
    """Compute SHA256 hash of a file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(
    cmd: list[str], 
    *, 
    check: bool = True, 
    capture: bool = False
) -> subprocess.CompletedProcess:
    """
    Execute command with auditability.
    
    Args:
        cmd: Command and arguments
        check: If True, raise on non-zero exit code. 
               Set to False for expected failures (like NEG control).
        capture: If True, capture stdout/stderr for logging.
    """
    print("\n$ " + " ".join(cmd))
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def safe_load_json(path: Path) -> dict:
    """Load JSON file, returning empty dict if file doesn't exist."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_first(d: Any, keys: list[str]) -> Optional[Any]:
    """
    Try multiple dotted paths and return the first found value.
    
    Example: find_first(data, ["C3_spectral.c3a_decoder.global", "C3.c3a.global"])
    """
    for k in keys:
        cur = d
        ok = True
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def copytree_overwrite(src: Path, dst: Path) -> None:
    """Copy directory tree, removing destination if it exists."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def ensure_spectrum_attrs(h5_path: Path, source_h5: Path) -> dict:
    """
    Ensure mixed spectrum.h5 has all required attrs.
    
    Strategy (robust, not fragile fixed list):
    1. Copy ALL attrs from source (e.g., ir_A) to mixed
    2. Overwrite only those that must reflect actual mixed data (n_delta)
    
    Returns dict with actions taken for audit.
    """
    import h5py
    
    actions = {"copied": [], "computed": [], "already_present": []}
    
    # Keys that must be recomputed from actual mixed data
    recompute_keys = {"n_delta"}
    
    with h5py.File(h5_path, "a") as h5:
        with h5py.File(source_h5, "r") as src:
            # Copy all attrs from source except those to recompute
            for k, v in src.attrs.items():
                if k in recompute_keys:
                    continue
                if k not in h5.attrs:
                    h5.attrs[k] = v
                    actions["copied"].append(k)
                else:
                    actions["already_present"].append(k)
        
        # Recompute from actual data
        actual_n_delta = h5["delta_uv"].shape[0]
        if "n_delta" not in h5.attrs or h5.attrs["n_delta"] != actual_n_delta:
            h5.attrs["n_delta"] = actual_n_delta
            actions["computed"].append(f"n_delta={actual_n_delta}")
        
        if "n_modes" not in h5.attrs:
            n_modes = h5["M2"].shape[1]
            h5.attrs["n_modes"] = n_modes
            actions["computed"].append(f"n_modes={n_modes}")
    
    return actions


def get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Experiment configuration with canonical defaults."""
    # Source runs
    run_a: str
    run_b: str
    run_mix: str

    # Derived runs for dictionary (to avoid overwriting outputs)
    run_mix_a: str
    run_mix_b: str

    # Experiment stage directory name
    exp_stage: str

    # Dataset generation parameters
    n_delta: int
    n_modes: int
    profiles: str
    delta_min: float
    delta_max: float
    seed: int

    # ir_A (eft_power) parameters
    a_family: str
    a_power_n: float
    a_map_mode: str
    a_alpha_min: float
    a_alpha_max: float

    # ir_B (symmetron) parameters
    b_family: str
    b_rho_crit: float
    b_map_mode: str
    b_alpha_min: float
    b_alpha_max: float

    # Dictionary common parameters
    k_features: int
    n_bootstrap: int

    # Negative control C3 configuration
    neg_c3_metric: str
    neg_c3_weights: str
    neg_c3_threshold: float

    # Positive control C3 configuration
    pos_c3_metric: str
    pos_c3_weights: str
    pos_c3_threshold: float
    pos_c3_adaptive: bool

    # Metadata
    variant_label: Optional[str]
    is_canonical: bool

    # Hypothesis check tolerance
    c3ab_tolerance: float


def parse_args() -> Config:
    """Parse command line arguments with canonical defaults."""
    ap = argparse.ArgumentParser(
        description="BASURIN Exp 03: C3 metric sensitivity (v{})".format(__version__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Canonical run (defaults from Exp 03 document):
  python 05_exp03_c3_metric_sensitivity.py

  # Exploratory run with different threshold (creates exp03_mytag/):
  python 05_exp03_c3_metric_sensitivity.py --variant-label mytag --pos-threshold 0.04
"""
    )
    
    # Run names
    ap.add_argument("--run-a", default="ir_A", 
                    help="Run name for source A (eft_power)")
    ap.add_argument("--run-b", default="ir_B", 
                    help="Run name for source B (symmetron)")
    ap.add_argument("--run-mix", default="ir_mix", 
                    help="Run name for mixed dataset")

    # Dictionary parameters
    ap.add_argument("--k-features", type=int, default=CANONICAL_K_FEATURES,
                    help=f"Number of ratio features (canonical: {CANONICAL_K_FEATURES})")
    ap.add_argument("--n-bootstrap", type=int, default=CANONICAL_N_BOOTSTRAP,
                    help=f"Bootstrap replicas, 0=deterministic (canonical: {CANONICAL_N_BOOTSTRAP})")

    # Dataset generation
    ap.add_argument("--n-delta", type=int, default=80,
                    help="Number of Delta samples")
    ap.add_argument("--n-modes", type=int, default=5,
                    help="Number of modes/observables")
    ap.add_argument("--profiles", default=CANONICAL_PROFILES,
                    help=f"Comma-separated profiles (canonical: {CANONICAL_PROFILES})")
    ap.add_argument("--delta-min", type=float, default=1.55)
    ap.add_argument("--delta-max", type=float, default=5.50)
    ap.add_argument("--seed", type=int, default=42)

    # ir_A parameters
    ap.add_argument("--a-alpha-min", type=float, default=-0.05)
    ap.add_argument("--a-alpha-max", type=float, default=0.05)
    ap.add_argument("--a-power-n", type=float, default=1.0)
    ap.add_argument("--a-map-mode", default="linear", choices=["linear", "quad"])

    # ir_B parameters
    ap.add_argument("--b-alpha-min", type=float, default=-0.08)
    ap.add_argument("--b-alpha-max", type=float, default=0.08)
    ap.add_argument("--b-rho-crit", type=float, default=4.0)
    ap.add_argument("--b-map-mode", default="quad", choices=["linear", "quad"])

    # C3 thresholds
    ap.add_argument("--neg-threshold", type=float, default=CANONICAL_NEG_THRESHOLD,
                    help=f"C3 threshold for negative control (canonical: {CANONICAL_NEG_THRESHOLD})")
    ap.add_argument("--pos-threshold", type=float, default=CANONICAL_POS_THRESHOLD,
                    help=f"C3 threshold for positive control (canonical: {CANONICAL_POS_THRESHOLD})")

    # Hypothesis check
    ap.add_argument("--c3ab-tolerance", type=float, default=0.01,
                    help="Tolerance for |C3a-C3b| check in positive control")

    # Variant/non-canonical handling
    ap.add_argument("--variant-label", type=str, default=None,
                    help="Label for non-canonical run. Creates exp03_<label>/ instead of exp03/")
    ap.add_argument("--non-canonical", action="store_true",
                    help="Explicitly mark as non-canonical (auto-detected if params differ)")

    args = ap.parse_args()

    # Determine if this is a canonical run
    is_canonical = (
        args.neg_threshold == CANONICAL_NEG_THRESHOLD and
        args.pos_threshold == CANONICAL_POS_THRESHOLD and
        args.profiles == CANONICAL_PROFILES and
        args.k_features == CANONICAL_K_FEATURES and
        args.variant_label is None and
        not args.non_canonical
    )

    # Determine stage directory name
    if args.variant_label:
        exp_stage = f"exp03_{args.variant_label}"
    elif not is_canonical:
        # Auto-generate label for non-canonical runs without explicit label
        exp_stage = "exp03_noncanonical"
    else:
        exp_stage = "exp03"

    if not is_canonical and args.variant_label is None:
        print("WARNING: Non-canonical parameters detected. Use --variant-label to avoid overwriting canonical stage.")

    return Config(
        run_a=args.run_a,
        run_b=args.run_b,
        run_mix=args.run_mix,
        run_mix_a=f"{args.run_mix}__A",
        run_mix_b=f"{args.run_mix}__B",
        exp_stage=exp_stage,
        n_delta=args.n_delta,
        n_modes=args.n_modes,
        profiles=args.profiles,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        seed=args.seed,
        a_family="eft_power",
        a_power_n=args.a_power_n,
        a_map_mode=args.a_map_mode,
        a_alpha_min=args.a_alpha_min,
        a_alpha_max=args.a_alpha_max,
        b_family="symmetron",
        b_rho_crit=args.b_rho_crit,
        b_map_mode=args.b_map_mode,
        b_alpha_min=args.b_alpha_min,
        b_alpha_max=args.b_alpha_max,
        k_features=args.k_features,
        n_bootstrap=args.n_bootstrap,
        neg_c3_metric="rmse",
        neg_c3_weights="none",
        neg_c3_threshold=args.neg_threshold,
        pos_c3_metric="rmse_log",
        pos_c3_weights="inv_n4",
        pos_c3_threshold=args.pos_threshold,
        pos_c3_adaptive=True,
        variant_label=args.variant_label,
        is_canonical=is_canonical,
        c3ab_tolerance=args.c3ab_tolerance,
    )


# =============================================================================
# Metrics extraction
# =============================================================================

def extract_metrics(v: dict) -> dict:
    """
    Extract C3 metrics from validation.json.
    
    Uses multiple path options to handle potential schema variations.
    Returns N/A (None) for missing fields, never invents values.
    """
    return {
        "C2_cv_rmse": find_first(v, [
            "C2_consistency.cv_rmse", 
            "C2.cv_rmse"
        ]),
        "C3_status": find_first(v, [
            "C3_spectral.status", 
            "overall.C3_status"
        ]),
        "C3_failure_mode": find_first(v, [
            "C3_spectral.failure_mode", 
            "overall.C3_failure_mode"
        ]),
        "C3a_global": find_first(v, [
            "C3_spectral.c3a_decoder.global"
        ]),
        "C3b_global": find_first(v, [
            "C3_spectral.c3b_cycle.global"
        ]),
        "threshold_user": find_first(v, [
            "C3_spectral.threshold.user"
        ]),
        "threshold_adaptive": find_first(v, [
            "C3_spectral.threshold.adaptive"
        ]),
        "threshold_effective": find_first(v, [
            "C3_spectral.threshold.effective"
        ]),
        "noise_floor_median_sigma": find_first(v, [
            "C3_spectral.noise_floor.median_sigma"
        ]),
    }


def check_hypothesis(
    m_neg: dict, 
    m_pos: dict, 
    c3ab_tol: float,
    neg_evidence_exists: bool,
    pos_evidence_exists: bool
) -> dict:
    """
    Check hypothesis with robust criteria.
    
    Negative control must:
      - C3_status == "FAIL"
      - C3_failure_mode == "DECODER_MISMATCH" (if present in JSON)
    
    Positive control must:
      - C3_status == "PASS"
      - |C3a - C3b| < c3ab_tol (consistency check)
    
    Both must have evidence (validation.json) to be valid.
    """
    # Evidence checks
    neg_evidence_ok = neg_evidence_exists
    pos_evidence_ok = pos_evidence_exists

    # Negative control checks
    neg_status = m_neg.get("C3_status")
    neg_mode = m_neg.get("C3_failure_mode")
    neg_status_ok = neg_status == "FAIL"
    # failure_mode check: OK if DECODER_MISMATCH or if field is missing (N/A)
    neg_mode_ok = neg_mode == "DECODER_MISMATCH" or neg_mode is None
    neg_ok = neg_evidence_ok and neg_status_ok and neg_mode_ok

    # Positive control checks
    pos_status = m_pos.get("C3_status")
    pos_status_ok = pos_status == "PASS"
    
    c3a = m_pos.get("C3a_global")
    c3b = m_pos.get("C3b_global")
    if c3a is not None and c3b is not None:
        c3ab_diff = abs(c3a - c3b)
        c3ab_ok = c3ab_diff < c3ab_tol
    else:
        c3ab_diff = None
        c3ab_ok = False  # Can't verify without data
    
    pos_ok = pos_evidence_ok and pos_status_ok and c3ab_ok

    # Sensitivity demonstration (secondary result)
    c3a_neg = m_neg.get("C3a_global")
    c3a_pos = m_pos.get("C3a_global")
    sensitivity_demonstrated = (
        c3a_neg is not None and 
        c3a_pos is not None and 
        c3a_pos < c3a_neg
    )
    if sensitivity_demonstrated and c3a_neg > 0:
        sensitivity_reduction_pct = (c3a_neg - c3a_pos) / c3a_neg * 100
    else:
        sensitivity_reduction_pct = None

    return {
        "negative_control": {
            "evidence_exists": neg_evidence_ok,
            "status_ok": neg_status_ok,
            "status_observed": neg_status,
            "failure_mode_ok": neg_mode_ok,
            "failure_mode_observed": neg_mode,
            "overall_ok": neg_ok,
        },
        "positive_control": {
            "evidence_exists": pos_evidence_ok,
            "status_ok": pos_status_ok,
            "status_observed": pos_status,
            "c3ab_diff": c3ab_diff,
            "c3ab_tolerance": c3ab_tol,
            "c3ab_ok": c3ab_ok,
            "overall_ok": pos_ok,
        },
        "hypothesis_confirmed": neg_ok and pos_ok,
        "sensitivity_demonstrated": sensitivity_demonstrated,
        "sensitivity_reduction_pct": sensitivity_reduction_pct,
    }


# =============================================================================
# Main execution
# =============================================================================

def main() -> int:
    cfg = parse_args()
    runs = Path("runs")

    # Header
    print("=" * 70)
    print(f"BASURIN Exp 03: C3 metric sensitivity (v{__version__})")
    print(f"  Timestamp: {utc_now_iso()}")
    print(f"  Stage:     {cfg.exp_stage}")
    if cfg.is_canonical:
        print("  Mode:      ✓ CANONICAL (defaults from Exp 03 document)")
    else:
        print("  Mode:      ⚠️  NON-CANONICAL")
        if cfg.variant_label:
            print(f"  Label:     {cfg.variant_label}")
    print("=" * 70)

    # --- Step 1: Generate ir_A if missing ---
    a_spec = runs / cfg.run_a / "spectrum" / "spectrum.h5"
    if not a_spec.exists():
        print(f"\n[1/6] Generating {cfg.run_a} (family={cfg.a_family})...")
        run_cmd([
            sys.executable, "01_genera_neutrino_sandbox.py",
            "--run", cfg.run_a,
            "--family", cfg.a_family,
            "--n-delta", str(cfg.n_delta),
            "--n-modes", str(cfg.n_modes),
            "--profiles", cfg.profiles,
            "--delta-min", str(cfg.delta_min),
            "--delta-max", str(cfg.delta_max),
            "--map-mode", cfg.a_map_mode,
            "--alpha-min", str(cfg.a_alpha_min),
            "--alpha-max", str(cfg.a_alpha_max),
            "--power-n", str(cfg.a_power_n),
            "--noise-rel", "0.0",
            "--seed", str(cfg.seed),
        ])
    else:
        print(f"\n[1/6] {cfg.run_a} exists, skipping.")

    # --- Step 2: Generate ir_B if missing ---
    b_spec = runs / cfg.run_b / "spectrum" / "spectrum.h5"
    if not b_spec.exists():
        print(f"\n[2/6] Generating {cfg.run_b} (family={cfg.b_family})...")
        run_cmd([
            sys.executable, "01_genera_neutrino_sandbox.py",
            "--run", cfg.run_b,
            "--family", cfg.b_family,
            "--n-delta", str(cfg.n_delta),
            "--n-modes", str(cfg.n_modes),
            "--profiles", cfg.profiles,
            "--delta-min", str(cfg.delta_min),
            "--delta-max", str(cfg.delta_max),
            "--map-mode", cfg.b_map_mode,
            "--alpha-min", str(cfg.b_alpha_min),
            "--alpha-max", str(cfg.b_alpha_max),
            "--rho-crit", str(cfg.b_rho_crit),
            "--noise-rel", "0.0",
            "--seed", str(cfg.seed),
        ])
    else:
        print(f"\n[2/6] {cfg.run_b} exists, skipping.")

    # --- Step 3: Mix to ir_mix if missing ---
    mix_spec = runs / cfg.run_mix / "spectrum" / "spectrum.h5"
    if not mix_spec.exists():
        print(f"\n[3/6] Mixing {cfg.run_a} + {cfg.run_b} -> {cfg.run_mix}...")
        run_cmd([
            sys.executable, "01_mix_spectra.py",
            "--run-out", cfg.run_mix,
            "--run-a", cfg.run_a,
            "--run-b", cfg.run_b,
        ])
    else:
        print(f"\n[3/6] {cfg.run_mix} exists, skipping.")

    # Ensure HDF5 attrs are complete
    # NOTE: 01_mix_spectra.py may not copy all attrs. This step ensures
    # the mixed spectrum has all required attrs by copying from source.
    print("       Ensuring HDF5 attrs...")
    attr_actions = ensure_spectrum_attrs(mix_spec, a_spec)
    if attr_actions["copied"]:
        print(f"       Copied attrs: {attr_actions['copied']}")
    if attr_actions["computed"]:
        print(f"       Computed: {attr_actions['computed']}")

    # --- Step 4: Create derived runs ---
    print(f"\n[4/6] Creating derived runs {cfg.run_mix_a} and {cfg.run_mix_b}...")
    src_spectrum_dir = runs / cfg.run_mix / "spectrum"
    dst_a_spectrum_dir = runs / cfg.run_mix_a / "spectrum"
    dst_b_spectrum_dir = runs / cfg.run_mix_b / "spectrum"
    ensure_dir(dst_a_spectrum_dir.parent)
    ensure_dir(dst_b_spectrum_dir.parent)
    copytree_overwrite(src_spectrum_dir, dst_a_spectrum_dir)
    copytree_overwrite(src_spectrum_dir, dst_b_spectrum_dir)

    # --- Step 5: Run dictionary A (NEGATIVE control) ---
    print(f"\n[5/6] Running {cfg.run_mix_a} (NEGATIVE control)...")
    print(f"       metric={cfg.neg_c3_metric}, weights={cfg.neg_c3_weights}, threshold={cfg.neg_c3_threshold}")
    
    # NOTE: 04_diccionario.py returns exit code 1 when C3 fails.
    # For negative control, this is EXPECTED. We use check=False to continue.
    result_a = run_cmd([
        sys.executable, "04_diccionario.py",
        "--run", cfg.run_mix_a,
        "--enable-c3",
        "--k-features", str(cfg.k_features),
        "--n-bootstrap", str(cfg.n_bootstrap),
        "--c3-metric", cfg.neg_c3_metric,
        "--c3-weights", cfg.neg_c3_weights,
        "--c3-threshold", str(cfg.neg_c3_threshold),
    ], check=False, capture=True)

    # --- Step 6: Run dictionary B (POSITIVE control) ---
    print(f"\n[6/6] Running {cfg.run_mix_b} (POSITIVE control)...")
    print(f"       metric={cfg.pos_c3_metric}, weights={cfg.pos_c3_weights}, threshold={cfg.pos_c3_threshold}, adaptive={cfg.pos_c3_adaptive}")
    
    cmd_b = [
        sys.executable, "04_diccionario.py",
        "--run", cfg.run_mix_b,
        "--enable-c3",
        "--k-features", str(cfg.k_features),
        "--n-bootstrap", str(cfg.n_bootstrap),
        "--c3-metric", cfg.pos_c3_metric,
        "--c3-weights", cfg.pos_c3_weights,
        "--c3-threshold", str(cfg.pos_c3_threshold),
    ]
    if cfg.pos_c3_adaptive:
        cmd_b.append("--c3-adaptive-threshold")
    
    # NOTE: Also use check=False for positive control to ensure we capture
    # evidence even if it unexpectedly fails.
    result_b = run_cmd(cmd_b, check=False, capture=True)

    # --- Extract metrics ---
    print("\n" + "=" * 70)
    print("Extracting metrics...")
    print("=" * 70)

    val_a_path = runs / cfg.run_mix_a / "dictionary" / "validation.json"
    val_b_path = runs / cfg.run_mix_b / "dictionary" / "validation.json"
    
    val_a_exists = val_a_path.exists()
    val_b_exists = val_b_path.exists()
    
    val_a = safe_load_json(val_a_path)
    val_b = safe_load_json(val_b_path)

    m_a = extract_metrics(val_a)
    m_b = extract_metrics(val_b)

    # Check for missing evidence (hard failure)
    if not val_a_exists:
        print(f"ERROR: validation.json missing for A: {val_a_path}")
    if not val_b_exists:
        print(f"ERROR: validation.json missing for B: {val_b_path}")

    # --- Write auditable stage ---
    exp_dir = runs / cfg.run_mix / cfg.exp_stage
    out_dir = exp_dir / "outputs"
    ensure_dir(out_dir)

    # Write logs
    (out_dir / "log_A.txt").write_text(
        f"=== COMMAND ===\nreturncode: {result_a.returncode}\n\n"
        f"=== STDOUT ===\n{result_a.stdout}\n\n"
        f"=== STDERR ===\n{result_a.stderr}",
        encoding="utf-8"
    )
    (out_dir / "log_B.txt").write_text(
        f"=== COMMAND ===\nreturncode: {result_b.returncode}\n\n"
        f"=== STDOUT ===\n{result_b.stdout}\n\n"
        f"=== STDERR ===\n{result_b.stderr}",
        encoding="utf-8"
    )

    # Copy evidence files
    def copy_if_exists(src: Path, dst_name: str) -> tuple[Optional[str], Optional[str]]:
        """Copy file if exists, return (relative_path, sha256) or (None, None)."""
        if src.exists():
            dst = out_dir / dst_name
            shutil.copy2(src, dst)
            return dst_name, sha256_file(dst)
        return None, None

    copied = {}
    hashes = {}
    
    for name, src in [
        ("validation_A.json", val_a_path),
        ("validation_B.json", val_b_path),
        ("dict_stage_summary_A.json", runs / cfg.run_mix_a / "dictionary" / "stage_summary.json"),
        ("dict_stage_summary_B.json", runs / cfg.run_mix_b / "dictionary" / "stage_summary.json"),
    ]:
        rel, h = copy_if_exists(src, name)
        if rel:
            copied[name] = f"outputs/{rel}"
            hashes[f"outputs/{name}"] = h

    # Add logs to hashes
    for log_name in ["log_A.txt", "log_B.txt"]:
        h = sha256_file(out_dir / log_name)
        if h:
            hashes[f"outputs/{log_name}"] = h

    # Hypothesis check
    hyp_check = check_hypothesis(
        m_a, m_b, 
        cfg.c3ab_tolerance,
        val_a_exists, 
        val_b_exists
    )

    # Input hash
    input_hash = sha256_file(mix_spec)

    # Git commit
    git_commit = get_git_commit()

    # --- Write metrics.json ---
    metrics = {
        "experiment": "Exp03_C3_metric_sensitivity",
        "version": __version__,
        "timestamp_utc": utc_now_iso(),
        "run_mix": cfg.run_mix,
        "is_canonical": cfg.is_canonical,
        "variant_label": cfg.variant_label,
        "canonical_thresholds": {
            "neg": CANONICAL_NEG_THRESHOLD,
            "pos": CANONICAL_POS_THRESHOLD,
        },
        "derived_runs": {
            "A": cfg.run_mix_a,
            "B": cfg.run_mix_b,
        },
        "negative_control": {
            "description": "Expected FAIL: rmse + none + fixed threshold",
            "config": {
                "c3_metric": cfg.neg_c3_metric,
                "c3_weights": cfg.neg_c3_weights,
                "c3_threshold": cfg.neg_c3_threshold,
                "adaptive": False,
            },
            "evidence_exists": val_a_exists,
            "returncode": result_a.returncode,
            "metrics": m_a,
        },
        "positive_control": {
            "description": "Expected PASS: rmse_log + inv_n4 + adaptive threshold",
            "config": {
                "c3_metric": cfg.pos_c3_metric,
                "c3_weights": cfg.pos_c3_weights,
                "c3_threshold": cfg.pos_c3_threshold,
                "adaptive": cfg.pos_c3_adaptive,
            },
            "evidence_exists": val_b_exists,
            "returncode": result_b.returncode,
            "metrics": m_b,
        },
        "hypothesis_check": hyp_check,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    hashes["outputs/metrics.json"] = sha256_file(metrics_path)

    # --- Write stage_summary.json ---
    stage_summary = {
        "stage": cfg.exp_stage,
        "script": "05_exp03_c3_metric_sensitivity.py",
        "version": __version__,
        "timestamp_utc": utc_now_iso(),
        "is_canonical": cfg.is_canonical,
        "variant_label": cfg.variant_label,
        "params": asdict(cfg),
        "canonical_reference": {
            "neg_threshold": CANONICAL_NEG_THRESHOLD,
            "pos_threshold": CANONICAL_POS_THRESHOLD,
            "profiles": CANONICAL_PROFILES,
            "k_features": CANONICAL_K_FEATURES,
        },
        "inputs": {
            "spectrum": str(mix_spec.as_posix()),
            "sha256": input_hash,
        },
        "derived_runs": {
            "A": cfg.run_mix_a,
            "B": cfg.run_mix_b,
        },
        "outputs": {
            "directory": str(exp_dir.as_posix()),
        },
        "hashes": hashes,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "git_commit": git_commit,
        },
        "hypotheses": {
            "negative_expected": {
                "C3_status": "FAIL",
                "C3_failure_mode": "DECODER_MISMATCH",
            },
            "positive_expected": {
                "C3_status": "PASS",
                "C3ab_diff": f"< {cfg.c3ab_tolerance}",
            },
        },
        "observed": {
            "negative": m_a,
            "positive": m_b,
        },
        "hypothesis_check": hyp_check,
        "falsifiability": {
            "experiment_broken_if": [
                "negative_control.C3_status != FAIL",
                "negative_control.C3_failure_mode != DECODER_MISMATCH (if present)",
                f"positive_control.C3_status != PASS (with canonical threshold {CANONICAL_POS_THRESHOLD})",
                "validation.json missing for either control",
                "input spectrum hash changes without justification",
            ],
        },
        "attr_actions": attr_actions,
    }
    summary_path = exp_dir / "stage_summary.json"
    summary_path.write_text(json.dumps(stage_summary, indent=2), encoding="utf-8")

    # --- Write manifest.json ---
    manifest = {
        "stage": cfg.exp_stage,
        "run": cfg.run_mix,
        "version": __version__,
        "is_canonical": cfg.is_canonical,
        "timestamp_utc": utc_now_iso(),
        "artifacts": {
            "stage_summary.json": "stage_summary.json",
            "manifest.json": "manifest.json",
            **{k: k for k in hashes.keys()},
        },
        "hashes": hashes,
    }
    manifest_path = exp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    if not cfg.is_canonical:
        print("⚠️  NON-CANONICAL RUN" + (f" (label: {cfg.variant_label})" if cfg.variant_label else ""))
    print("=" * 70)

    print(f"\nNEGATIVE control (expected FAIL):")
    print(f"  Evidence:       {'✓' if val_a_exists else '✗ MISSING'}")
    print(f"  C3 status:      {m_a.get('C3_status', 'N/A')}")
    print(f"  C3 failure:     {m_a.get('C3_failure_mode', 'N/A')}")
    print(f"  C3a global:     {m_a.get('C3a_global', 'N/A')}")
    print(f"  C3b global:     {m_a.get('C3b_global', 'N/A')}")
    print(f"  threshold eff:  {m_a.get('threshold_effective', 'N/A')}")
    print(f"  noise floor:    {m_a.get('noise_floor_median_sigma', 'N/A')}")

    print(f"\nPOSITIVE control (expected PASS):")
    print(f"  Evidence:       {'✓' if val_b_exists else '✗ MISSING'}")
    print(f"  C3 status:      {m_b.get('C3_status', 'N/A')}")
    print(f"  C3 failure:     {m_b.get('C3_failure_mode', 'N/A')}")
    print(f"  C3a global:     {m_b.get('C3a_global', 'N/A')}")
    print(f"  C3b global:     {m_b.get('C3b_global', 'N/A')}")
    print(f"  threshold eff:  {m_b.get('threshold_effective', 'N/A')}")
    print(f"  noise floor:    {m_b.get('noise_floor_median_sigma', 'N/A')}")

    # Verdict
    print("\n" + "-" * 70)
    neg_ok = hyp_check["negative_control"]["overall_ok"]
    pos_ok = hyp_check["positive_control"]["overall_ok"]
    sens_ok = hyp_check["sensitivity_demonstrated"]

    if neg_ok and pos_ok:
        print("✓ HYPOTHESIS CONFIRMED: Negative FAIL, Positive PASS")
    else:
        print("✗ HYPOTHESIS NOT FULLY CONFIRMED:")
        if not val_a_exists:
            print("  - NEGATIVE: evidence_missing (validation.json not found)")
        elif not neg_ok:
            print(f"  - NEGATIVE: status={m_a.get('C3_status')}, mode={m_a.get('C3_failure_mode')}")
        if not val_b_exists:
            print("  - POSITIVE: evidence_missing (validation.json not found)")
        elif not pos_ok:
            print(f"  - POSITIVE: status={m_b.get('C3_status')}")
            if m_b.get('C3_status') == "FAIL":
                c3a = m_b.get('C3a_global')
                thr = m_b.get('threshold_effective')
                if c3a is not None and thr is not None:
                    print(f"    (C3a={c3a:.4f} > threshold={thr})")

    if sens_ok:
        reduction = hyp_check.get("sensitivity_reduction_pct")
        c3a_neg = m_a.get('C3a_global', 0)
        c3a_pos = m_b.get('C3a_global', 0)
        print(f"\n✓ SENSITIVITY DEMONSTRATED: C3a {c3a_neg:.4f} → {c3a_pos:.4f} ({reduction:.1f}% reduction)")

    print("-" * 70)
    print(f"\nStage written to: {exp_dir}")

    # Return code:
    # 0 = hypothesis confirmed OR sensitivity demonstrated (valid scientific result)
    # 1 = neither (something is broken)
    # 2 = evidence missing (hard failure)
    if not val_a_exists or not val_b_exists:
        return 2
    if neg_ok and pos_ok:
        return 0
    if sens_ok:
        return 0  # Sensitivity demonstration is a valid result
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
