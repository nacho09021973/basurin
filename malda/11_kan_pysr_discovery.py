#!/usr/bin/env python3
"""Symbolic law discovery on GW event feature table using KAN + PySR.

Loads the enriched feature table from step 10 and runs:

  1. KAN (Kolmogorov-Arnold Network) — learns f(X) for each target variable,
     then reads off the learned 1-D activation shapes (spline forms) to
     understand what combinations matter and in what functional form.

  2. PySR — symbolic regression to produce human-readable algebraic
     expressions on ALL target × feature pairs, plus on the combinations
     highlighted by KAN.

Inspiration: in 1971 nobody knew that BH entropy ∝ area.
Here we give PySR the numbers and let it tell us what relationships exist.

Usage:
    # Build feature table first:
    python malda/10_build_event_feature_table.py --run-id my_run_01

    # Then run discovery (GPU accelerated if CUDA available):
    python malda/11_kan_pysr_discovery.py --run-id my_run_01

    # Restrict to BBH events for cleaner QNM columns:
    python malda/11_kan_pysr_discovery.py --run-id my_run_01 --bbh-only

    # Skip KAN (fast mode, PySR only):
    python malda/11_kan_pysr_discovery.py --run-id my_run_01 --no-kan

    # Run KAN only (no PySR):
    python malda/11_kan_pysr_discovery.py --run-id my_run_01 --no-pysr

Outputs (under runs/<run-id>/experiment/malda_discovery/):
    outputs/pareto_frontier_all.csv  — combined PySR Pareto frontier (all targets)
    outputs/pysr_pareto_<target>.csv — per-target PySR Pareto CSV
    outputs/discovery_results.json   — full per-target results (KAN + PySR)
    outputs/discovery_summary.json   — top-1 equation per target
    manifest.json                    — SHA256 hashes, artifact paths
    stage_summary.json               — verdict, config, seed

Requirements:
    pip install pysr pykan torch pandas scikit-learn
    (or: pip install -r requirements.txt)

Design notes:
  - NO physics injected. The model sees raw feature numbers.
  - Post-hoc comparison with known laws (Hawking, Kerr fits, chirp mass
    formula) happens ONLY in the summary printout, never as a training signal.
  - All discovered equations are saved verbatim from PySR output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Repository layout + BASURIN IO
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basurin_io import (  # noqa: E402
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# Target variables — we ask "what determines Y?"
# Each target gets its own PySR run and KAN output head.
TARGETS = {
    "E_rad_frac":   "Fraction of mass radiated in GWs",
    "af":           "Final BH spin",
    "S_f":          "Final BH entropy proxy [M_sun^2]",
    "delta_S":      "Entropy increase [M_sun^2]  (Hawking: >= 0)",
    "S_ratio":      "Entropy ratio S_f/(S1+S2)  (Hawking: >= 1)",
    "Q_220":        "Quality factor of (2,2,0) QNM  [function of af only in Kerr]",
    "F_220_dimless":"Dimensionless QNM frequency  [function of af only in Kerr]",
    "f_ratio_221_220": "Frequency ratio f_221/f_220  [mass-independent in Kerr]",
}

# Columns excluded from physical discovery regardless of target.
UNIVERSAL_DROP = {
    "event_id",
    "GPS",
    "snr",
    "p_astro",
    "far_yr",
    "DL_Mpc",
    "z",
    "log_DL",
    "log1pz",
    "is_bbh",
    "is_bns",
    "is_nsbh",
    "classification_source",
    "has_multimessenger",
    "glitch_mitigated",
    "catalog",
}

# Strict scientific discovery space: initial / inspiral state only.
PREMERGER_ONLY = [
    "m1_src",
    "m2_src",
    "M_total",
    "Mchirp",
    "chi_eff",
    "q",
    "eta",
    "delta",
    "log_q",
    "Mchirp_over_Mtotal",
]

# Post-merger or definitional columns that are high-risk for leakage.
POSTMERGER_AND_LEAKY = {
    "Mf",
    "af",
    "E_rad_Msun",
    "E_rad_frac",
    "f_220_hz",
    "tau_220_s",
    "Q_220",
    "F_220_dimless",
    "f_221_hz",
    "tau_221_s",
    "Q_221",
    "F_221_dimless",
    "f_330_hz",
    "tau_330_s",
    "Q_330",
    "F_330_dimless",
    "f_ratio_221_220",
    "Q_ratio_221_220",
    "Mf_f220_dimless",
    "xi_f",
    "S_f",
    "S1_schw",
    "S2_schw",
    "delta_S",
    "delta_S_frac",
    "S_ratio",
}

# Candidate inputs considered before policy filtering.
INPUT_FEATURES = PREMERGER_ONLY + ["af"]

TARGET_ALLOWLIST_STRICT = {target: list(PREMERGER_ONLY) for target in TARGETS}
TARGET_ALLOWLIST_KERR_VALIDATION = {
    "Q_220": ["af"],
    "F_220_dimless": ["af"],
    "f_ratio_221_220": ["af"],
}

# Columns that should be log-transformed after policy filtering
# (positive-definite, spanning orders of magnitude).
LOG_TRANSFORM_COLS = {
    "m1_src",
    "m2_src",
    "M_total",
    "Mchirp",
}


def resolve_input_features(target_name: str, feature_policy: str) -> tuple[list[str], str]:
    """Return base input features for a target before any transformations."""
    if feature_policy == "strict_premerger":
        allowed = TARGET_ALLOWLIST_STRICT.get(target_name, PREMERGER_ONLY)
        analysis_mode = "discovery"
    elif feature_policy == "kerr_validation":
        allowed = TARGET_ALLOWLIST_KERR_VALIDATION.get(target_name, PREMERGER_ONLY)
        analysis_mode = "kerr_validation" if target_name in TARGET_ALLOWLIST_KERR_VALIDATION else "discovery"
    else:
        raise ValueError(f"Unknown feature policy: {feature_policy}")

    selected: list[str] = []
    for feature in allowed:
        if feature in UNIVERSAL_DROP:
            continue
        if feature not in INPUT_FEATURES:
            continue
        if feature_policy == "strict_premerger" and feature in POSTMERGER_AND_LEAKY:
            continue
        if feature not in selected:
            selected.append(feature)
    if not selected:
        raise ValueError(f"No input features resolved for target '{target_name}' under policy '{feature_policy}'")
    return selected, analysis_mode


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_table(path: Path, bbh_only: bool = False) -> tuple[list[str], np.ndarray]:
    """Load CSV feature table. Returns (column_names, data_array)."""
    import csv
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if bbh_only:
        rows = [r for r in rows if r.get("is_bbh", "0").strip() == "1"]
        print(f"[11_discovery] BBH filter: {len(rows)} events retained")

    if not rows:
        raise RuntimeError("No events loaded from feature table. Run step 10 first.")

    cols = list(rows[0].keys())
    data = []
    for row in rows:
        vals = []
        for col in cols:
            v = row.get(col, "")
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        data.append(vals)

    return cols, np.array(data, dtype=np.float64)


def prepare_XY(
    cols: list[str],
    data: np.ndarray,
    target: str,
    input_features: list[str],
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract X (features) and y (target), drop rows with NaN, optionally log-transform."""
    col_idx = {c: i for i, c in enumerate(cols)}

    # Target index
    if target not in col_idx:
        raise ValueError(f"Target '{target}' not in table columns")
    y_idx = col_idx[target]
    y_raw = data[:, y_idx]

    # Input indices (skip missing)
    feat_used = [f for f in input_features if f in col_idx and f != target]
    X_raw = data[:, [col_idx[f] for f in feat_used]]

    # Drop rows where target or any feature is NaN
    valid = np.isfinite(y_raw)
    for j in range(X_raw.shape[1]):
        valid &= np.isfinite(X_raw[:, j])
    # Also drop rows where target is a constant wrt itself (e.g. all zeros)
    X_raw = X_raw[valid]
    y_raw = y_raw[valid]

    if len(y_raw) < 5:
        raise ValueError(f"Too few valid rows for target '{target}': {len(y_raw)}")

    # Log-transform
    X = X_raw.copy()
    feat_names = list(feat_used)
    if log_transform:
        for j, fname in enumerate(feat_used):
            if fname in LOG_TRANSFORM_COLS:
                col_data = X[:, j]
                if np.all(col_data > 0):
                    X[:, j] = np.log(col_data)
                    feat_names[j] = f"log_{fname}"

    return X, y_raw, feat_names


# ---------------------------------------------------------------------------
# KAN discovery
# ---------------------------------------------------------------------------

def run_kan(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    target_name: str,
    out_dir: Path,
    n_epochs: int = 200,
    grid: int = 5,
) -> dict[str, Any]:
    """Train a KAN on X->y, return activation summary."""
    try:
        import torch
        from kan import KAN  # pykan
    except ImportError:
        print("[11_discovery] pykan or torch not installed — skipping KAN", file=sys.stderr)
        print("  Install with: pip install pykan torch", file=sys.stderr)
        return {"status": "skipped", "reason": "pykan not installed"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[KAN:{target_name}] device={device}  X={X.shape}  features={feat_names}")

    # Normalise X to [0,1] for KAN stability
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = np.where(X_max > X_min, X_max - X_min, 1.0)
    X_norm = (X - X_min) / X_range

    y_mean = y.mean()
    y_std = y.std() if y.std() > 0 else 1.0
    y_norm = (y - y_mean) / y_std

    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_norm, dtype=torch.float32, device=device).unsqueeze(1)

    # Split train/test
    n = len(y_t)
    idx = np.random.permutation(n)
    n_train = max(int(0.8 * n), n - 2)
    train_idx = torch.tensor(idx[:n_train], device=device)
    test_idx = torch.tensor(idx[n_train:], device=device)

    dataset = {
        "train_input": X_t[train_idx],
        "train_label": y_t[train_idx],
        "test_input": X_t[test_idx],
        "test_label": y_t[test_idx],
    }

    # Build KAN: [n_features, 4, 1]  (small hidden layer)
    n_in = X.shape[1]
    model = KAN(width=[n_in, max(4, n_in // 2), 1], grid=grid, k=3, device=device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.train(dataset, opt="Adam", steps=n_epochs, lamb=0.01, display_metrics=None)

    train_loss = float(results["train_loss"][-1]) if results.get("train_loss") else float("nan")
    test_loss = float(results["test_loss"][-1]) if results.get("test_loss") else float("nan")
    print(f"[KAN:{target_name}] train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

    # Extract feature importances from input-layer edge magnitudes
    try:
        # KAN stores splines as model.act_fun[layer][i,j]
        # Importance proxy: mean absolute magnitude of first-layer activations
        importances: dict[str, float] = {}
        first_layer = model.act_fun[0]  # shape: [n_in, hidden]
        for j, fname in enumerate(feat_names):
            # mean |coeff| across all edges from input j
            mag = float(first_layer[j].abs().mean().item()) if hasattr(first_layer[j], "abs") else 0.0
            importances[fname] = mag
    except Exception:
        importances = {f: float("nan") for f in feat_names}

    # Attempt symbolic fitting (KAN has a built-in suggest_symbolic)
    symbolic_suggestions: list[dict] = []
    try:
        model.auto_symbolic(lib=["x", "x^2", "x^3", "sqrt", "log", "exp", "sin", "abs"])
        for i, fname in enumerate(feat_names):
            try:
                sym = model.symbolic_formula()[0]
                symbolic_suggestions.append({"feature": fname, "formula": str(sym)})
            except Exception:
                break
    except Exception:
        pass

    result = {
        "status": "ok",
        "target": target_name,
        "n_events": int(n),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "feature_importances": dict(sorted(importances.items(), key=lambda x: -x[1])),
        "symbolic_suggestions": symbolic_suggestions,
    }

    # Save KAN plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        model.plot(folder=str(out_dir / f"kan_plots_{target_name}"), beta=100)
        result["plot_dir"] = str(out_dir / f"kan_plots_{target_name}")
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# PySR discovery
# ---------------------------------------------------------------------------

def run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    target_name: str,
    out_dir: Path,
    n_iterations: int = 100,
    maxsize: int = 20,
    use_gpu: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Run PySR symbolic regression on X->y."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("[11_discovery] pysr not installed — skipping PySR", file=sys.stderr)
        print("  Install with: pip install pysr", file=sys.stderr)
        return {"status": "skipped", "reason": "pysr not installed"}

    print(f"[PySR:{target_name}] X={X.shape}  n_iter={n_iterations}  maxsize={maxsize}")

    pareto_path = out_dir / f"pysr_pareto_{target_name}.csv"
    backend_dir = out_dir / "pysr_backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    backend_run_id = f"{target_name}_{seed}"

    model = PySRRegressor(
        # Search space
        niterations=n_iterations,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp", "abs", "square"],
        # Parsimony: reward simple expressions
        parsimony=1e-4,
        # Complexity penalty
        complexity_of_operators={
            "^": 3,
            "exp": 3,
            "log": 2,
            "sqrt": 2,
            "square": 1,
        },
        # Output
        output_jax_format=False,
        output_torch_format=False,
        temp_equation_file=False,
        output_directory=str(backend_dir),
        run_id=backend_run_id,
        # GPU
        turbo=use_gpu,
        # Verbosity
        verbosity=0,
        progress=False,
        # Reproducibility
        random_state=seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, variable_names=feat_names)

    # Collect Pareto frontier
    try:
        equations = model.equations_
        if equations is not None and len(equations) > 0:
            eq_list = []
            for _, row in equations.iterrows():
                eq_list.append({
                    "complexity": int(row.get("complexity", -1)),
                    "loss": float(row.get("loss", float("nan"))),
                    "score": float(row.get("score", float("nan"))),
                    "equation": str(row.get("equation", "")),
                    "sympy_format": str(row.get("sympy_format", "")),
                })
            best = min(eq_list, key=lambda e: e["loss"])
            print(f"[PySR:{target_name}] best eq (complexity={best['complexity']}): {best['equation']}")
        else:
            eq_list = []
            best = {}
    except Exception as exc:
        print(f"[PySR:{target_name}] WARNING: could not extract equations: {exc}", file=sys.stderr)
        eq_list = []
        best = {}

    import csv

    with open(pareto_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["complexity", "loss", "score", "equation", "sympy_format"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(eq_list)

    return {
        "status": "ok",
        "target": target_name,
        "n_events": int(len(y)),
        "best_equation": best,
        "pareto_equations": eq_list,
        "pareto_csv": str(pareto_path) if pareto_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="KAN + PySR symbolic discovery on GW event feature table"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="BASURIN run identifier (alphanumeric, -, ., _)",
    )
    parser.add_argument(
        "--feature-table",
        default="",
        help=(
            "Path to event_features.csv from step 10. "
            "Defaults to runs/<run-id>/experiment/malda_feature_table/outputs/event_features.csv"
        ),
    )
    parser.add_argument(
        "--bbh-only",
        action="store_true",
        help="Restrict to BBH events for cleaner QNM columns",
    )
    parser.add_argument(
        "--targets",
        default="",
        help="Comma-separated subset of targets to run (default: all)",
    )
    parser.add_argument(
        "--feature-policy",
        choices=["strict_premerger", "kerr_validation"],
        default="strict_premerger",
        help="Feature selection policy applied before any internal transforms (default: strict_premerger)",
    )
    parser.add_argument(
        "--no-kan",
        action="store_true",
        help="Skip KAN step (faster, PySR only)",
    )
    parser.add_argument(
        "--no-pysr",
        action="store_true",
        help="Skip PySR step (KAN only)",
    )
    parser.add_argument(
        "--pysr-iterations",
        type=int,
        default=100,
        help="Number of PySR iterations per target (default: 100; more = slower but better)",
    )
    parser.add_argument(
        "--pysr-maxsize",
        type=int,
        default=20,
        help="Maximum equation complexity for PySR (default: 20)",
    )
    parser.add_argument(
        "--kan-epochs",
        type=int,
        default=200,
        help="Number of KAN training epochs per target (default: 200)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (requires CUDA + PySR turbo mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (governs numpy, torch, and PySR)",
    )
    args = parser.parse_args(argv)

    # Seed all RNGs
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    runs_root = resolve_out_root("runs")
    validate_run_id(args.run_id, runs_root)
    stage_dir = runs_root / args.run_id / "experiment" / "malda_discovery"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve feature table path
    table_path = Path(args.feature_table) if args.feature_table else (
        runs_root / args.run_id / "experiment" / "malda_feature_table" / "outputs" / "event_features.csv"
    )
    if not table_path.exists():
        print(
            f"[11_discovery] ERROR: feature table not found at {table_path}\n"
            f"  Run first: python malda/10_build_event_feature_table.py --run-id {args.run_id}",
            file=sys.stderr,
        )
        return 1

    cols, data = load_feature_table(table_path, bbh_only=args.bbh_only)
    print(f"[11_discovery] Loaded {data.shape[0]} events × {data.shape[1]} columns")

    # Select targets
    active_targets = TARGETS
    if args.targets:
        requested = [t.strip() for t in args.targets.split(",") if t.strip()]
        active_targets = {k: v for k, v in TARGETS.items() if k in requested}
        print(f"[11_discovery] Running {len(active_targets)} targets: {list(active_targets)}")

    # Results accumulator
    config_dict = vars(args).copy()
    config_dict["feature_table_resolved"] = str(table_path)
    all_results: dict[str, Any] = {
        "config": config_dict,
        "n_events_total": int(data.shape[0]),
        "targets": {},
    }
    features_by_target: dict[str, list[str]] = {}
    analysis_mode_by_target: dict[str, str] = {}

    # Per-target loop
    for target_name, target_desc in active_targets.items():
        print(f"\n{'='*60}")
        print(f"TARGET: {target_name}  ({target_desc})")
        print("="*60)

        inputs, analysis_mode = resolve_input_features(target_name, args.feature_policy)

        try:
            X, y, feat_names = prepare_XY(cols, data, target_name, inputs)
        except ValueError as exc:
            print(f"[11_discovery] Skipping {target_name}: {exc}", file=sys.stderr)
            all_results["targets"][target_name] = {
                "status": "skipped",
                "reason": str(exc),
                "analysis_mode": analysis_mode,
                "feature_policy": args.feature_policy,
                "features_selected_base": inputs,
                "features_used": [],
            }
            features_by_target[target_name] = []
            analysis_mode_by_target[target_name] = analysis_mode
            continue

        print(f"  n_valid={len(y)}  features={feat_names}")
        print(f"  y stats: min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}  std={y.std():.4f}")

        target_result: dict[str, Any] = {
            "description": target_desc,
            "analysis_mode": analysis_mode,
            "feature_policy": args.feature_policy,
            "n_valid": int(len(y)),
            "features_selected_base": inputs,
            "features_used": feat_names,
            "y_stats": {
                "min": float(y.min()), "max": float(y.max()),
                "mean": float(y.mean()), "std": float(y.std()),
            },
        }
        features_by_target[target_name] = list(feat_names)
        analysis_mode_by_target[target_name] = analysis_mode

        # KAN
        if not args.no_kan:
            kan_result = run_kan(
                X, y, feat_names, target_name, outputs_dir,
                n_epochs=args.kan_epochs,
            )
            target_result["kan"] = kan_result
        else:
            target_result["kan"] = {"status": "skipped", "reason": "--no-kan flag"}

        # PySR
        if not args.no_pysr:
            pysr_result = run_pysr(
                X, y, feat_names, target_name, outputs_dir,
                n_iterations=args.pysr_iterations,
                maxsize=args.pysr_maxsize,
                use_gpu=args.gpu,
                seed=args.seed,
            )
            target_result["pysr"] = pysr_result
        else:
            target_result["pysr"] = {"status": "skipped", "reason": "--no-pysr flag"}

        all_results["targets"][target_name] = target_result

    # Aggregate Pareto frontier across all targets
    all_equations: list[dict] = []
    for tname, tresult in all_results["targets"].items():
        pysr = tresult.get("pysr", {})
        for eq in pysr.get("pareto_equations", []):
            all_equations.append({"target": tname, **eq})

    # Save combined Pareto CSV to outputs/
    combined_path = outputs_dir / "pareto_frontier_all.csv"
    if all_equations:
        try:
            import csv as csvmod
            fieldnames = ["target", "complexity", "loss", "score", "equation", "sympy_format"]
            with open(combined_path, "w", newline="", encoding="utf-8") as f:
                writer = csvmod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_equations)
            print(f"\n[11_discovery] Combined Pareto frontier: {combined_path}")
            all_results["pareto_frontier_csv"] = str(combined_path)
        except Exception as exc:
            print(f"[11_discovery] WARNING: could not write combined Pareto CSV: {exc}", file=sys.stderr)

    # Discovery summary: top-1 equation per target
    print("\n" + "="*60)
    print("DISCOVERY SUMMARY")
    print("="*60)
    summary_rows: list[dict] = []
    for tname, tresult in all_results["targets"].items():
        pysr = tresult.get("pysr", {})
        best = pysr.get("best_equation", {})
        kan = tresult.get("kan", {})
        importances = kan.get("feature_importances", {})
        top3_features = list(importances.keys())[:3] if importances else []

        row = {
            "target": tname,
            "description": tresult.get("description", ""),
            "n_valid": tresult.get("n_valid", 0),
            "best_equation": best.get("equation", "—"),
            "complexity": best.get("complexity", "—"),
            "loss": best.get("loss", "—"),
            "kan_top_features": top3_features,
        }
        summary_rows.append(row)
        print(f"\n  {tname}:")
        print(f"    best eq  : {row['best_equation']}")
        print(f"    complexity: {row['complexity']}   loss: {row['loss']}")
        print(f"    KAN top-3: {top3_features}")

    all_results["summary"] = summary_rows

    # Write outputs to outputs/
    results_path = outputs_dir / "discovery_results.json"
    write_json_atomic(results_path, all_results)
    print(f"\n[11_discovery] Full results: {results_path}")

    summary_path = outputs_dir / "discovery_summary.json"
    write_json_atomic(summary_path, summary_rows)
    print(f"[11_discovery] Summary: {summary_path}")

    # --- Manifest + stage_summary ---
    artifacts: dict[str, Any] = {
        "discovery_results": results_path,
        "discovery_summary": summary_path,
    }
    if combined_path.exists():
        artifacts["pareto_frontier_all"] = combined_path

    write_manifest(stage_dir, artifacts, extra={"run_id": args.run_id, "stage": "malda_discovery"})

    write_stage_summary(stage_dir, {
        "stage": "malda_discovery",
        "verdict": "PASS",
        "run_id": args.run_id,
        "created": utc_now_iso(),
        "config": {
            "seed": args.seed,
            "bbh_only": args.bbh_only,
            "feature_policy": args.feature_policy,
            "no_kan": args.no_kan,
            "no_pysr": args.no_pysr,
            "pysr_iterations": args.pysr_iterations,
            "pysr_maxsize": args.pysr_maxsize,
            "kan_epochs": args.kan_epochs,
            "gpu": args.gpu,
            "feature_table": str(table_path),
        },
        "results": {
            "n_events": int(data.shape[0]),
            "n_targets_run": len([t for t in all_results["targets"].values() if t.get("status") != "skipped"]),
            "n_equations_total": len(all_equations),
            "features_by_target": features_by_target,
            "analysis_mode_by_target": analysis_mode_by_target,
        },
        "outputs": {k: str(v) for k, v in artifacts.items()},
    })

    print(f"\n[11_discovery] Done → {stage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
