#!/usr/bin/env python3
"""B5-Z — Continuous Surface Emulator (BRUNETE port of E5-Z).

Gaussian Process Reconstruction over the physical parameter space of each
QG family.  Answers the referee question: "is there a hidden minimum between
your atlas grid nodes?"

Operates on all events in a BRUNETE classify run, either per-event or
aggregating across events.  Each event's per-family GP surfaces can be
compared directly.

Governance
----------
- Read-only on compatible_set.json per event subrun (all RUN_VALID=PASS).
- Writes only under runs/<classify_run_id>/experiment/continuous_emulator_<mode>/.
- Self-aborts if R² < 0.90 (SURFACE_UNLEARNABLE).
- External dependencies: numpy, scikit-learn, scipy.
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

from brunete.experiment.base_contract import (
    EVENT_RUN_GATES,
    GovernanceViolation,
    _write_json_atomic,
    ensure_experiment_dir,
    enumerate_event_runs,
    load_json,
    resolve_classify_run_dir,
    sha256_file,
    write_manifest,
)

SCHEMA_VERSION = "b5z-0.1"
EXPERIMENT_NAME = "continuous_emulator"
MIN_R2_THRESHOLD = 0.90
DEFAULT_GRID_RESOLUTION = 50

# Maps family → physical parameter keys to extract from geometry metadata
FAMILY_PARAMS: dict[str, list[str]] = {
    "edgb": ["chi", "zeta"],
    "dcs": ["chi", "zeta"],
    "kerr_newman": ["chi", "q_charge"],
    "kerr": ["chi"],
}

_FAMILY_ALIASES = {
    "EdGB": "edgb", "EDGB": "edgb", "einstein_dilaton_gauss_bonnet": "edgb",
    "dCS": "dcs", "DCS": "dcs", "dynamical_chern_simons": "dcs",
    "Kerr-Newman": "kerr_newman", "Kerr_Newman": "kerr_newman", "KN": "kerr_newman",
    "Kerr": "kerr", "GR_Kerr": "kerr", "gr_kerr": "kerr",
}


class SurfaceUnlearnable(GovernanceViolation):
    """Raised when GP cannot learn the surface (R² < threshold)."""


def _normalize_family(name: str) -> str:
    return _FAMILY_ALIASES.get(name, name.lower().replace("-", "_"))


def _extract_params(geometry: dict, family: str) -> dict[str, float] | None:
    norm_family = _normalize_family(family)
    param_keys = FAMILY_PARAMS.get(norm_family)
    if param_keys is None:
        return None
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
    if target == "d2":
        for key in ("d2", "mahalanobis_d2", "distance_squared"):
            val = geometry.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
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


def _fit_gp_surface(
    X: np.ndarray,
    y: np.ndarray,
    kernel_name: str = "Matern52",
) -> tuple[Any, float, np.ndarray]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

    if kernel_name == "Matern52":
        base = Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    elif kernel_name == "Matern32":
        base = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    elif kernel_name == "RBF":
        base = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10,
        normalize_y=True, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)

    n = len(y)
    loo_residuals = np.zeros(n)
    if n >= 4:
        loo_predictions = np.zeros(n)
        for i in range(n):
            X_tr = np.delete(X, i, axis=0)
            y_tr = np.delete(y, i)
            gp_loo = GaussianProcessRegressor(
                kernel=gp.kernel_, optimizer=None,
                normalize_y=True, random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_loo.fit(X_tr, y_tr)
            loo_predictions[i] = gp_loo.predict(X[i:i + 1])[0]
            loo_residuals[i] = y[i] - loo_predictions[i]
        ss_res = np.sum(loo_residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        y_pred = gp.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        loo_residuals = y - y_pred

    return gp, r2, loo_residuals


def _find_continuous_minimum(
    gp: Any,
    param_keys: list[str],
    grid_resolution: int = 200,
) -> tuple[dict[str, float], float, float]:
    """Find predicted minimum on a fine normalized grid."""
    if len(param_keys) == 1:
        X_fine = np.linspace(0, 1, grid_resolution).reshape(-1, 1)
    elif len(param_keys) == 2:
        g0, g1 = np.meshgrid(
            np.linspace(0, 1, grid_resolution),
            np.linspace(0, 1, grid_resolution),
            indexing="ij",
        )
        X_fine = np.column_stack([g0.ravel(), g1.ravel()])
    else:
        raise ValueError(f"Unsupported dimensionality: {len(param_keys)}")

    y_pred, y_std = gp.predict(X_fine, return_std=True)
    idx_min = int(np.argmin(y_pred))
    min_params_norm = {k: float(X_fine[idx_min, j]) for j, k in enumerate(param_keys)}
    return min_params_norm, float(y_pred[idx_min]), float(y_std[idx_min])


def _generate_surface_grid(
    gp: Any, param_keys: list[str], resolution: int,
) -> dict[str, Any]:
    if len(param_keys) == 1:
        X_fine = np.linspace(0, 1, resolution).reshape(-1, 1)
        y_pred, y_std = gp.predict(X_fine, return_std=True)
        return {
            "param_names": param_keys,
            "grid": {param_keys[0]: X_fine[:, 0].tolist()},
            "predicted_d2": y_pred.tolist(),
            "uncertainty": y_std.tolist(),
            "resolution": resolution,
        }
    elif len(param_keys) == 2:
        g0, g1 = np.meshgrid(
            np.linspace(0, 1, resolution),
            np.linspace(0, 1, resolution),
            indexing="ij",
        )
        X_fine = np.column_stack([g0.ravel(), g1.ravel()])
        y_pred, y_std = gp.predict(X_fine, return_std=True)
        return {
            "param_names": param_keys,
            "grid": {
                param_keys[0]: np.linspace(0, 1, resolution).tolist(),
                param_keys[1]: np.linspace(0, 1, resolution).tolist(),
            },
            "predicted_d2": y_pred.reshape(resolution, resolution).tolist(),
            "uncertainty": y_std.reshape(resolution, resolution).tolist(),
            "resolution": resolution,
        }
    else:
        raise ValueError(f"Unsupported dimensionality: {len(param_keys)}")


def _hidden_minimum_confidence(
    gp: Any, X_train: np.ndarray, y_train: np.ndarray, param_keys: list[str],
    grid_resolution: int = 200,
) -> dict[str, Any]:
    from scipy.stats import norm as normal_dist

    n_params = len(param_keys)
    if n_params == 1:
        X_fine = np.linspace(0, 1, grid_resolution).reshape(-1, 1)
    elif n_params == 2:
        g0, g1 = np.meshgrid(
            np.linspace(0, 1, grid_resolution),
            np.linspace(0, 1, grid_resolution), indexing="ij",
        )
        X_fine = np.column_stack([g0.ravel(), g1.ravel()])
    else:
        return {"confidence_level": "UNKNOWN", "reason": "dimensionality > 2"}

    y_pred, y_std = gp.predict(X_fine, return_std=True)
    gp_min = float(np.min(y_pred))
    mean_std = float(np.mean(y_std))
    max_std = float(np.max(y_std))
    margin = max(mean_std, 1e-6)
    threshold = gp_min - margin
    z_scores = (threshold - y_pred) / np.maximum(y_std, 1e-10)
    max_prob_hidden = float(np.max(normal_dist.cdf(z_scores)))
    confidence = 1.0 - max_prob_hidden

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
            f"There is a {confidence * 100:.1f}% confidence that no minimum "
            f"deeper than {gp_min - margin:.3f} (GP min − 1σ) exists between "
            f"the atlas grid nodes for this family."
        ),
    }


def emulate_family_for_event(
    event_id: str,
    event_run_dir: Path,
    family: str,
    target: str = "d2",
    kernel: str = "Matern52",
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    use_all_geometries: bool = True,
) -> dict[str, Any]:
    """Run GP emulator for a single family in a single event subrun."""
    norm_family = _normalize_family(family)
    param_keys = FAMILY_PARAMS.get(norm_family)
    if param_keys is None:
        raise ValueError(f"Unknown family {family!r}. Supported: {list(FAMILY_PARAMS.keys())}")

    cs_path = event_run_dir / EVENT_RUN_GATES["compatible_set"]
    if not cs_path.exists():
        return {
            "schema_version": SCHEMA_VERSION, "event_id": event_id,
            "family": norm_family, "status": "NO_COMPATIBLE_SET",
        }

    cs = load_json(cs_path)
    if use_all_geometries:
        geometries = cs.get("ranked_all", []) or cs.get("compatible_geometries", [])
        if not geometries:
            geometries = cs if isinstance(cs, list) else cs.get("geometries", [])
    else:
        geometries = cs.get("compatible_geometries", [])
        if not geometries:
            geometries = cs if isinstance(cs, list) else cs.get("geometries", [])

    # Filter to family
    family_entries = []
    for g in geometries:
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
                "geometry_id": str(g.get("geometry_id", g.get("id", "?"))),
                "params": params,
                "target": target_val,
            })

    if len(family_entries) < 3:
        return {
            "schema_version": SCHEMA_VERSION, "event_id": event_id,
            "family": norm_family, "status": "INSUFFICIENT_DATA",
            "n_geometries": len(family_entries),
            "message": f"Need ≥3 geometries for GPR, found {len(family_entries)}",
        }

    X = np.array([[e["params"][k] for k in param_keys] for e in family_entries])
    y = np.array([e["target"] for e in family_entries])

    # Normalize X to [0,1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range

    gp, r2, loo_residuals = _fit_gp_surface(X_norm, y, kernel_name=kernel)

    if r2 < MIN_R2_THRESHOLD:
        return {
            "schema_version": SCHEMA_VERSION, "event_id": event_id,
            "family": norm_family, "status": "SURFACE_UNLEARNABLE",
            "r2_score": round(float(r2), 4),
            "n_geometries": len(family_entries),
            "kernel_used": f"{kernel} + WhiteKernel",
            "message": f"R²={r2:.3f} < {MIN_R2_THRESHOLD}.",
        }

    best_idx = int(np.argmin(y))
    discrete_best = family_entries[best_idx]

    min_params_norm, min_val, min_std = _find_continuous_minimum(gp, param_keys, 200)
    min_params = {
        k: round(float(min_params_norm[k] * X_range[j] + X_min[j]), 6)
        for j, k in enumerate(param_keys)
    }

    surface = _generate_surface_grid(gp, param_keys, resolution=grid_resolution)
    for j, k in enumerate(param_keys):
        surface["grid"][k] = [
            round(float(v * X_range[j] + X_min[j]), 6) for v in surface["grid"][k]
        ]

    subgrid_improvement = min_val < discrete_best["target"]
    improvement_pct = (
        (discrete_best["target"] - min_val) / abs(discrete_best["target"]) * 100
        if discrete_best["target"] != 0 else 0.0
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "family": norm_family,
        "status": "SUCCESS",
        "kernel_used": f"{kernel} + WhiteKernel",
        "kernel_optimized": str(gp.kernel_),
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
        "input_hash": sha256_file(cs_path),
        "no_hidden_minimum_confidence": _hidden_minimum_confidence(
            gp, X_norm, y, param_keys, grid_resolution=200
        ),
    }


def emulate_classify_run(
    classify_run_id: str,
    mode: str = "220",
    families: list[str] | None = None,
    target: str = "d2",
    kernel: str = "Matern52",
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run B5-Z for all events and all families in a classify run."""
    if families is None:
        families = list(FAMILY_PARAMS.keys())

    event_run_map = enumerate_event_runs(classify_run_id, mode=mode, runs_root=runs_root)
    input_hashes: dict[str, str] = {}
    per_event: list[dict] = []

    for event_id, event_run_dir in sorted(event_run_map.items()):
        cs_path = event_run_dir / EVENT_RUN_GATES["compatible_set"]
        if cs_path.exists():
            input_hashes[event_id] = sha256_file(cs_path)

        per_family: dict[str, Any] = {}
        for fam in families:
            per_family[fam] = emulate_family_for_event(
                event_id=event_id,
                event_run_dir=event_run_dir,
                family=fam,
                target=target,
                kernel=kernel,
                grid_resolution=grid_resolution,
            )
        per_event.append({"event_id": event_id, "families": per_family})

    return {
        "schema_version": SCHEMA_VERSION,
        "classify_run_id": classify_run_id,
        "mode": mode,
        "families": families,
        "target_variable": target,
        "kernel": kernel,
        "grid_resolution": grid_resolution,
        "n_events": len(per_event),
        "per_event": per_event,
        "input_hashes": input_hashes,
    }


def run_b5z(
    classify_run_id: str,
    mode: str = "220",
    families: list[str] | None = None,
    target: str = "d2",
    kernel: str = "Matern52",
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    runs_root: str | Path | None = None,
) -> Path:
    run_dir = resolve_classify_run_dir(classify_run_id, runs_root)
    exp_dir = ensure_experiment_dir(run_dir, f"{EXPERIMENT_NAME}_{mode}")

    result = emulate_classify_run(
        classify_run_id, mode=mode, families=families,
        target=target, kernel=kernel,
        grid_resolution=grid_resolution, runs_root=runs_root,
    )

    out_path = exp_dir / "emulator_manifest.json"
    _write_json_atomic(out_path, {k: v for k, v in result.items() if k != "per_event"})

    for event_result in result["per_event"]:
        event_id = event_result["event_id"]
        for fam, fam_result in event_result["families"].items():
            if fam_result.get("status") == "SUCCESS":
                surface_path = exp_dir / f"gpr_surface_{event_id}_{fam}.json"
                _write_json_atomic(surface_path, fam_result.get("surface_grid", {}))
                minima_path = exp_dir / f"predicted_minima_{event_id}_{fam}.json"
                _write_json_atomic(minima_path, {
                    "event_id": event_id,
                    "family": fam,
                    "discrete_best": fam_result.get("discrete_best_geometry"),
                    "continuous_minimum": fam_result.get("continuous_predicted_minimum"),
                    "r2": fam_result.get("r2_score"),
                    "subgrid_improvement": fam_result.get("subgrid_improvement"),
                    "no_hidden_minimum_confidence": fam_result.get("no_hidden_minimum_confidence"),
                })

    write_manifest(exp_dir, result["input_hashes"], extra={
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "kernel": kernel,
        "families": result["families"],
    })
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B5-Z: GPR continuous surface emulator over all events in a classify run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classify-run", required=True)
    ap.add_argument("--mode", choices=["220", "221"], default="220")
    ap.add_argument("--families", nargs="+", default=None,
                    help="Families to emulate (default: all)")
    ap.add_argument("--target", choices=["d2", "delta_lnL"], default="d2")
    ap.add_argument("--kernel", choices=["Matern52", "Matern32", "RBF"], default="Matern52")
    ap.add_argument("--grid-resolution", type=int, default=DEFAULT_GRID_RESOLUTION)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    result = emulate_classify_run(
        args.classify_run, mode=args.mode,
        families=args.families, target=args.target,
        kernel=args.kernel, grid_resolution=args.grid_resolution,
        runs_root=args.runs_root,
    )

    if args.dry_run:
        summary = {
            "classify_run_id": result["classify_run_id"],
            "mode": result["mode"],
            "n_events": result["n_events"],
            "families": result["families"],
        }
        for ev in result["per_event"]:
            ev_id = ev["event_id"]
            for fam, fam_r in ev["families"].items():
                status = fam_r.get("status", "?")
                if status == "SUCCESS":
                    conf = fam_r.get("no_hidden_minimum_confidence", {})
                    print(f"  {ev_id} / {fam}: {status} "
                          f"R²={fam_r.get('r2_score')} "
                          f"conf={conf.get('confidence_level', '?')}")
                else:
                    print(f"  {ev_id} / {fam}: {status}")
        return 0

    out_path = run_b5z(
        args.classify_run, mode=args.mode,
        families=args.families, target=args.target,
        kernel=args.kernel, grid_resolution=args.grid_resolution,
        runs_root=args.runs_root,
    )
    print(f"B5-Z written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
