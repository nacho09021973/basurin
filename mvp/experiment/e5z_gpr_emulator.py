#!/usr/bin/env python3
"""E5-Z — Continuous Surface Emulator (Gaussian Process Reconstruction).

The definitive cure against discretization bias ("the missed minimum between
grid nodes").  Fits a Gaussian Process over the physical parameter space of
a family (e.g., spin χ × coupling ζ for EdGB) using d² or delta_lnL evaluated
at atlas nodes.

The emulator does NOT filter — it interpolates.  It reconstructs the continuous
"compatibility valley", predicts the exact mathematical coordinate of the true
minimum, and provides Bayesian uncertainty bands on the interpolation.

Key output:  ``predicted_minima.json`` contains the inferred continuous minimum
per family, with GPR uncertainty — the answer to "is there a hidden minimum
between your grid nodes?".

Governance:
  - Read-only on estimates.json + compatible_set ranked_all (RUN_VALID=PASS).
  - Writes only under runs/<run_id>/experiment/continuous_emulator/.
  - Self-aborts if R² < 0.90 ("surface_unlearnable").
  - Never promoted to core (statistical fiction, not numerical relativity).
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mvp.experiment.base_contract import (
    REQUIRED_CANONICAL_GATES,
    GovernanceViolation,
    _write_json_atomic,
    ensure_experiment_dir,
    load_json,
    sha256_file,
    validate_and_load_run,
    write_manifest,
)

SCHEMA_VERSION = "e5z-0.1"
EXPERIMENT_NAME = "continuous_emulator"
MIN_R2_THRESHOLD = 0.90
DEFAULT_GRID_RESOLUTION = 50  # N×N fine grid for surface reconstruction


# ── Family parameter extraction ─────────────────────────────────────────────

# Maps family name → list of parameter keys to extract from metadata
FAMILY_PARAMS: dict[str, list[str]] = {
    "edgb": ["chi", "zeta"],
    "dcs": ["chi", "zeta"],
    "kerr_newman": ["chi", "q_charge"],
    "kerr": ["chi"],
}

# Aliases for robustness
_FAMILY_ALIASES = {
    "EdGB": "edgb",
    "EDGB": "edgb",
    "einstein_dilaton_gauss_bonnet": "edgb",
    "dCS": "dcs",
    "DCS": "dcs",
    "dynamical_chern_simons": "dcs",
    "Kerr-Newman": "kerr_newman",
    "Kerr_Newman": "kerr_newman",
    "KN": "kerr_newman",
    "Kerr": "kerr",
    "GR_Kerr": "kerr",
    "gr_kerr": "kerr",
}


def _normalize_family(name: str) -> str:
    """Normalize family name to canonical form."""
    return _FAMILY_ALIASES.get(name, name.lower().replace("-", "_"))


def _extract_params(geometry: dict, family: str) -> dict[str, float] | None:
    """Extract physical parameters for a geometry entry.

    Returns dict of param_name → value, or None if extraction fails.
    """
    norm_family = _normalize_family(family)
    param_keys = FAMILY_PARAMS.get(norm_family)
    if param_keys is None:
        return None

    # Try metadata first, then top-level
    meta = geometry.get("metadata", {})
    params = {}
    for key in param_keys:
        val = meta.get(key, geometry.get(key))
        if val is None:
            return None
        try:
            params[key] = float(val)
        except (ValueError, TypeError):
            return None
    return params


def _extract_target(geometry: dict, target: str = "d2") -> float | None:
    """Extract the target variable (d2 or delta_lnL) from a geometry entry."""
    # Try multiple field names
    if target == "d2":
        for key in ("d2", "mahalanobis_d2", "distance_squared"):
            val = geometry.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        # Compute from distance if available
        dist = geometry.get("distance")
        if dist is not None:
            try:
                return float(dist) ** 2
            except (ValueError, TypeError):
                pass
    elif target == "delta_lnL":
        for key in ("delta_lnL", "delta_log_likelihood"):
            val = geometry.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    return None


# ── Core GP Emulator ────────────────────────────────────────────────────────

class SurfaceUnlearnable(GovernanceViolation):
    """Raised when the GP cannot learn the surface (R² < threshold)."""


def _fit_gp_surface(
    X: np.ndarray,
    y: np.ndarray,
    kernel_name: str = "Matern52",
) -> tuple[Any, float, np.ndarray]:
    """Fit GP to data.  Returns (gp_model, r2_score, loo_residuals).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    kernel_name : "Matern52", "RBF", or "Matern32"

    Returns
    -------
    gp : fitted GaussianProcessRegressor
    r2 : R² score on training data (leave-one-out cross-validated)
    loo_residuals : array of LOO residuals
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        RBF,
        WhiteKernel,
    )

    # Build kernel
    if kernel_name == "Matern52":
        base_kernel = Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    elif kernel_name == "Matern32":
        base_kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    elif kernel_name == "RBF":
        base_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel + WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42,  # determinism
    )

    # Fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)

    # Leave-one-out cross-validation for R²
    n = len(y)
    loo_predictions = np.zeros(n)
    loo_residuals = np.zeros(n)

    if n >= 4:
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            gp_loo = GaussianProcessRegressor(
                kernel=gp.kernel_,  # use optimized kernel
                optimizer=None,  # skip re-optimization for speed
                normalize_y=True,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_loo.fit(X_train, y_train)
            loo_predictions[i] = gp_loo.predict(X[i : i + 1])[0]
            loo_residuals[i] = y[i] - loo_predictions[i]

        ss_res = np.sum(loo_residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        # Too few points for LOO, use training R²
        y_pred = gp.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        loo_residuals = y - y_pred

    return gp, r2, loo_residuals


def _find_continuous_minimum(
    gp: Any,
    param_bounds: dict[str, tuple[float, float]],
    param_names: list[str],
    grid_resolution: int = 200,
) -> tuple[dict[str, float], float, float]:
    """Find the predicted minimum on a fine grid.

    Returns (optimal_params, predicted_d2, gpr_uncertainty).
    """
    # Create fine grid
    grids = []
    for name in param_names:
        lo, hi = param_bounds[name]
        grids.append(np.linspace(lo, hi, grid_resolution))

    if len(param_names) == 1:
        X_fine = grids[0].reshape(-1, 1)
    elif len(param_names) == 2:
        g0, g1 = np.meshgrid(grids[0], grids[1], indexing="ij")
        X_fine = np.column_stack([g0.ravel(), g1.ravel()])
    else:
        raise ValueError(f"Unsupported dimensionality: {len(param_names)}")

    y_pred, y_std = gp.predict(X_fine, return_std=True)

    # Find minimum
    idx_min = np.argmin(y_pred)
    min_params = {}
    for j, name in enumerate(param_names):
        min_params[name] = float(X_fine[idx_min, j])
    min_val = float(y_pred[idx_min])
    min_std = float(y_std[idx_min])

    return min_params, min_val, min_std


def _generate_surface_grid(
    gp: Any,
    param_bounds: dict[str, tuple[float, float]],
    param_names: list[str],
    resolution: int = DEFAULT_GRID_RESOLUTION,
) -> dict[str, Any]:
    """Generate a fine-grid surface for visualization."""
    grids = []
    for name in param_names:
        lo, hi = param_bounds[name]
        grids.append(np.linspace(lo, hi, resolution))

    if len(param_names) == 1:
        X_fine = grids[0].reshape(-1, 1)
        y_pred, y_std = gp.predict(X_fine, return_std=True)
        return {
            "param_names": param_names,
            "grid": {param_names[0]: grids[0].tolist()},
            "predicted_d2": y_pred.tolist(),
            "uncertainty": y_std.tolist(),
            "resolution": resolution,
        }
    elif len(param_names) == 2:
        g0, g1 = np.meshgrid(grids[0], grids[1], indexing="ij")
        X_fine = np.column_stack([g0.ravel(), g1.ravel()])
        y_pred, y_std = gp.predict(X_fine, return_std=True)
        return {
            "param_names": param_names,
            "grid": {
                param_names[0]: grids[0].tolist(),
                param_names[1]: grids[1].tolist(),
            },
            "predicted_d2": y_pred.reshape(resolution, resolution).tolist(),
            "uncertainty": y_std.reshape(resolution, resolution).tolist(),
            "resolution": resolution,
        }
    else:
        raise ValueError(f"Unsupported dimensionality: {len(param_names)}")


# ── High-level emulator ────────────────────────────────────────────────────

def emulate_family(
    run_id: str,
    family: str,
    target: str = "d2",
    kernel: str = "Matern52",
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    runs_root: str | Path | None = None,
    use_all_geometries: bool = True,
) -> dict[str, Any]:
    """Run GP emulator for a single family in a single run.

    Parameters
    ----------
    run_id : Run identifier (must have RUN_VALID=PASS).
    family : Family name (edgb, dcs, kerr_newman, kerr).
    target : Target variable ("d2" or "delta_lnL").
    kernel : GP kernel ("Matern52", "Matern32", "RBF").
    grid_resolution : Resolution of the fine output grid.
    runs_root : Override runs root directory.
    use_all_geometries : If True, use ranked_all (all evaluated geometries),
                         not just compatible ones.  This gives the GP more
                         training data including rejected geometries.
    """
    norm_family = _normalize_family(family)
    param_keys = FAMILY_PARAMS.get(norm_family)
    if param_keys is None:
        raise ValueError(
            f"Unknown family '{family}'. Supported: {list(FAMILY_PARAMS.keys())}"
        )

    run_dir, _ = validate_and_load_run(run_id, runs_root)
    cs_path = run_dir / REQUIRED_CANONICAL_GATES["compatible_set"]
    if not cs_path.exists():
        legacy_cs_path = run_dir / "s4_geometry_filter" / "compatible_set.json"
        if legacy_cs_path.exists():
            cs_path = legacy_cs_path
        else:
            raise FileNotFoundError(f"compatible_set.json missing: {cs_path}")

    cs = load_json(cs_path)

    # Extract geometries — prefer ranked_all (includes non-compatible) for more GP data
    if use_all_geometries:
        geometries = cs.get("ranked_all", [])
        if not geometries:
            geometries = cs.get("compatible_geometries", [])
            if not geometries:
                geometries = cs if isinstance(cs, list) else cs.get("geometries", [])
    else:
        geometries = cs.get("compatible_geometries", [])
        if not geometries:
            geometries = cs if isinstance(cs, list) else cs.get("geometries", [])

    # Filter to family and extract params + target
    family_entries = []
    for g in geometries:
        # Check family membership
        g_family = g.get("family", "")
        if not g_family:
            meta = g.get("metadata", {})
            g_family = meta.get("family", meta.get("theory", ""))
        if _normalize_family(g_family) != norm_family:
            continue

        params = _extract_params(g, family)
        target_val = _extract_target(g, target)
        if params is not None and target_val is not None and math.isfinite(target_val):
            family_entries.append({
                "geometry_id": g.get("geometry_id", g.get("id", "?")),
                "params": params,
                "target": target_val,
            })

    if len(family_entries) < 3:
        return {
            "schema_version": SCHEMA_VERSION,
            "family": norm_family,
            "status": "INSUFFICIENT_DATA",
            "n_geometries": len(family_entries),
            "message": f"Need ≥3 geometries for GPR, found {len(family_entries)}",
        }

    # Build arrays
    X = np.array([[e["params"][k] for k in param_keys] for e in family_entries])
    y = np.array([e["target"] for e in family_entries])

    # Normalize X to [0, 1] for better GP behavior
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # avoid division by zero
    X_norm = (X - X_min) / X_range

    # Fit GP
    gp, r2, loo_residuals = _fit_gp_surface(X_norm, y, kernel_name=kernel)

    if r2 < MIN_R2_THRESHOLD:
        return {
            "schema_version": SCHEMA_VERSION,
            "family": norm_family,
            "status": "SURFACE_UNLEARNABLE",
            "r2_score": round(float(r2), 4),
            "n_geometries": len(family_entries),
            "kernel_used": f"{kernel} + WhiteKernel",
            "message": f"R²={r2:.3f} < {MIN_R2_THRESHOLD}. Surface too irregular for reliable interpolation.",
        }

    # Find discrete best
    best_idx = int(np.argmin(y))
    discrete_best = family_entries[best_idx]

    # Find continuous minimum (on normalized grid, then denormalize)
    param_bounds_norm = {k: (0.0, 1.0) for k in param_keys}
    min_params_norm, min_val, min_std = _find_continuous_minimum(
        gp, param_bounds_norm, param_keys, grid_resolution=200
    )

    # Denormalize
    min_params = {}
    for j, k in enumerate(param_keys):
        min_params[k] = round(float(min_params_norm[k] * X_range[j] + X_min[j]), 6)

    # Generate surface grid (normalized, then denormalize coordinates)
    surface = _generate_surface_grid(gp, param_bounds_norm, param_keys, resolution=grid_resolution)
    # Replace grid coordinates with physical values
    for j, k in enumerate(param_keys):
        surface["grid"][k] = [
            round(float(v * X_range[j] + X_min[j]), 6) for v in surface["grid"][k]
        ]

    # Determine if continuous minimum improves on discrete
    subgrid_improvement = min_val < discrete_best["target"]

    # Compute improvement magnitude
    improvement_pct = (
        (discrete_best["target"] - min_val) / abs(discrete_best["target"]) * 100
        if discrete_best["target"] != 0
        else 0.0
    )

    # Input hashes
    input_hashes = {"compatible_set": sha256_file(cs_path)}
    est_path = run_dir / REQUIRED_CANONICAL_GATES["estimates"]
    if est_path.exists():
        input_hashes["estimates"] = sha256_file(est_path)

    # Kernel hyperparameters
    kernel_params = str(gp.kernel_)

    return {
        "schema_version": SCHEMA_VERSION,
        "family": norm_family,
        "status": "SUCCESS",
        "kernel_used": f"{kernel} + WhiteKernel",
        "kernel_optimized": kernel_params,
        "n_geometries": len(family_entries),
        "target_variable": target,
        "param_names": param_keys,
        "discrete_best_geometry": {
            "geometry_id": discrete_best["geometry_id"],
            "params": discrete_best["params"],
            f"measured_{target}": round(float(discrete_best["target"]), 6),
        },
        "continuous_predicted_minimum": {
            "params": min_params,
            f"interpolated_{target}": round(float(min_val), 6),
            "gpr_uncertainty": round(float(min_std), 6),
        },
        "subgrid_improvement": bool(subgrid_improvement),
        "improvement_percent": round(float(improvement_pct), 2),
        "r2_score": round(float(r2), 4),
        "loo_residuals": {
            "mean": round(float(np.mean(loo_residuals)), 6),
            "std": round(float(np.std(loo_residuals)), 6),
            "max_abs": round(float(np.max(np.abs(loo_residuals))), 6),
        },
        "surface_grid": surface,
        "input_hashes": input_hashes,
        "no_hidden_minimum_confidence": _hidden_minimum_confidence(
            gp, X_norm, y, param_keys, grid_resolution=200
        ),
    }


def _hidden_minimum_confidence(
    gp: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_keys: list[str],
    grid_resolution: int = 200,
) -> dict[str, Any]:
    """Compute confidence that no minimum deeper than the GP-predicted one is hidden.

    The GP already finds a continuous minimum (possibly sub-grid).  The question
    is: how confident are we that the *true* surface doesn't dip significantly
    below the GP-predicted minimum?  This is bounded by the worst-case
    predictive uncertainty across the parameter space.

    Strategy: find the GP-predicted minimum μ*, then compute for every point
    on a fine grid the probability P(f(x) < μ* - margin) where margin is
    set to the mean uncertainty.  High confidence means the GP surface is
    well-constrained everywhere.
    """
    # Generate fine grid
    n_params = len(param_keys)
    if n_params == 1:
        X_fine = np.linspace(0, 1, grid_resolution).reshape(-1, 1)
    elif n_params == 2:
        g0, g1 = np.meshgrid(
            np.linspace(0, 1, grid_resolution),
            np.linspace(0, 1, grid_resolution),
            indexing="ij",
        )
        X_fine = np.column_stack([g0.ravel(), g1.ravel()])
    else:
        return {"confidence_level": "UNKNOWN", "reason": "dimensionality > 2"}

    y_pred, y_std = gp.predict(X_fine, return_std=True)

    # GP-predicted minimum
    gp_min = float(np.min(y_pred))
    mean_std = float(np.mean(y_std))
    max_std = float(np.max(y_std))

    # Margin: how much deeper than the GP minimum would be "significantly hidden"
    # Use 1σ of the mean uncertainty as the threshold
    margin = max(mean_std, 1e-6)

    # For each grid point, P(true f(x) < gp_min - margin)
    from scipy.stats import norm as normal_dist

    threshold = gp_min - margin
    z_scores = (threshold - y_pred) / np.maximum(y_std, 1e-10)
    prob_deeper = normal_dist.cdf(z_scores)

    # Worst-case probability across all points
    max_prob_hidden = float(np.max(prob_deeper))
    confidence = 1.0 - max_prob_hidden

    # Also compute relative uncertainty: max_std / range of predictions
    pred_range = float(np.max(y_pred) - np.min(y_pred))
    relative_uncertainty = max_std / pred_range if pred_range > 0 else float("inf")

    if confidence >= 0.997:
        level = "VERY_HIGH (3σ)"
    elif confidence >= 0.95:
        level = "HIGH (2σ)"
    elif confidence >= 0.68:
        level = "MODERATE (1σ)"
    else:
        level = "LOW"

    return {
        "confidence_no_hidden_minimum": round(confidence, 6),
        "confidence_level": level,
        "max_hidden_probability": round(max_prob_hidden, 6),
        "gp_predicted_minimum": round(gp_min, 6),
        "mean_uncertainty": round(mean_std, 6),
        "max_uncertainty": round(max_std, 6),
        "relative_uncertainty": round(relative_uncertainty, 6),
        "interpretation": (
            f"There is a {confidence*100:.1f}% confidence that no minimum "
            f"deeper than {gp_min - margin:.3f} (GP min − 1σ) exists between "
            f"the atlas grid nodes for this family."
        ),
    }


# ── Multi-family convenience ────────────────────────────────────────────────

def emulate_all_families(
    run_id: str,
    families: list[str] | None = None,
    target: str = "d2",
    kernel: str = "Matern52",
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run GP emulator for all (or specified) families in a run."""
    if families is None:
        families = list(FAMILY_PARAMS.keys())

    results = {}
    for fam in families:
        results[fam] = emulate_family(
            run_id=run_id,
            family=fam,
            target=target,
            kernel=kernel,
            grid_resolution=grid_resolution,
            runs_root=runs_root,
        )

    # Collect predicted minima across families
    predicted_minima = {}
    for fam, res in results.items():
        if res.get("status") == "SUCCESS":
            predicted_minima[fam] = res["continuous_predicted_minimum"]

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "families_emulated": list(results.keys()),
        "predicted_minima": predicted_minima,
        "per_family_results": results,
    }


# ── Write outputs ───────────────────────────────────────────────────────────

def run_emulator(
    run_id: str,
    families: list[str] | None = None,
    target: str = "d2",
    kernel: str = "Matern52",
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full emulator pipeline: validate, fit, predict, write."""
    result = emulate_all_families(
        run_id=run_id,
        families=families,
        target=target,
        kernel=kernel,
        grid_resolution=grid_resolution,
        runs_root=runs_root,
    )

    if dry_run:
        # Print without surface grids (too large for stdout)
        summary = dict(result)
        for fam, res in summary.get("per_family_results", {}).items():
            if "surface_grid" in res:
                res = dict(res)
                res["surface_grid"] = "<omitted for dry-run>"
                summary["per_family_results"][fam] = res
        print(json.dumps(summary, indent=2, default=str))
        return result

    run_dir, _ = validate_and_load_run(run_id, runs_root)
    out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    # Predicted minima (headline result)
    _write_json_atomic(out_dir / "predicted_minima.json", result["predicted_minima"])

    # Per-family surfaces (for visualization)
    for fam, res in result["per_family_results"].items():
        if res.get("status") == "SUCCESS":
            _write_json_atomic(out_dir / f"gpr_surface_{fam}.json", res["surface_grid"])
            _write_json_atomic(out_dir / f"validation_residuals_{fam}.json", res["loo_residuals"])

    # Manifest
    all_hashes = {}
    for res in result["per_family_results"].values():
        if "input_hashes" in res:
            all_hashes.update(res["input_hashes"])

    _write_json_atomic(out_dir / "emulator_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "kernel": kernel,
        "target": target,
        "grid_resolution": grid_resolution,
        "families": list(result["per_family_results"].keys()),
        "per_family_status": {
            fam: res.get("status", "UNKNOWN")
            for fam, res in result["per_family_results"].items()
        },
        "input_hashes": all_hashes,
    })

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5-Z: Continuous Surface Emulator (Gaussian Process Reconstruction)"
    )
    parser.add_argument("--run-id", required=True, help="Run ID (must be RUN_VALID=PASS)")
    parser.add_argument("--families", nargs="*", default=None,
                        help="Families to emulate (default: all known)")
    parser.add_argument("--target", choices=["d2", "delta_lnL"], default="d2",
                        help="Target variable for GP (default: d2)")
    parser.add_argument("--kernel", choices=["Matern52", "Matern32", "RBF"], default="Matern52",
                        help="GP kernel (default: Matern52)")
    parser.add_argument("--grid-resolution", type=int, default=DEFAULT_GRID_RESOLUTION,
                        help=f"Output grid resolution (default: {DEFAULT_GRID_RESOLUTION})")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_emulator(
        run_id=args.run_id,
        families=args.families,
        target=args.target,
        kernel=args.kernel,
        grid_resolution=args.grid_resolution,
        runs_root=args.runs_root,
        dry_run=args.dry_run,
    )

    # Summary
    for fam, res in result.get("per_family_results", {}).items():
        status = res.get("status", "?")
        if status == "SUCCESS":
            disc = res["discrete_best_geometry"]
            cont = res["continuous_predicted_minimum"]
            conf = res["no_hidden_minimum_confidence"]
            print(f"\n{fam.upper()}:")
            print(f"  Discrete best: {disc['geometry_id']} → d²={disc.get('measured_d2', '?')}")
            print(f"  GP predicted:  {cont['params']} → d²={cont.get('interpolated_d2', '?')} ± {cont.get('gpr_uncertainty', '?')}")
            print(f"  R² = {res['r2_score']}, improvement = {res['improvement_percent']:.1f}%")
            print(f"  No hidden minimum: {conf['confidence_level']} ({conf['confidence_no_hidden_minimum']*100:.1f}%)")
        else:
            print(f"\n{fam.upper()}: {status} — {res.get('message', '')}")


if __name__ == "__main__":
    main()
