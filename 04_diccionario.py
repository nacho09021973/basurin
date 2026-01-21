#!/usr/bin/env python3
"""
Bloque C: Diccionario holográfico inverso λ_SL → Δ

v1.4.0 - Diagnóstico causal C3 con separación C3a/C3b, pesos, noise floor adaptativo

Aprende la relación inversa: dado un espectro de masas M_n², predice Δ_UV.
Usa features invariantes de escala (ratios r_n = M_n²/M_0²).

Uso:
    python 04_diccionario.py --run mi_experimento
    python 04_diccionario.py --run mi_experimento --enable-c3
    python 04_diccionario.py --run mi_experimento --enable-c3 --c3-weights inv_n4 --c3-adaptive-threshold on

Salida:
    runs/<run>/dictionary/
        ├── manifest.json
        ├── stage_summary.json
        └── outputs/
            ├── dictionary.h5          (modelo, coeficientes, métricas)
            ├── atlas.json             (clustering de teorías efectivas)
            ├── ising_comparison.json  (contraste con Δ_σ, Δ_ε)
            └── validation.json        (contratos C1, C2, C3, C5)

Contratos:
    C1: Compatibilidad puntual con Ising (τ_Δ=0.02 o 2σ)
    C2: Consistencia interna (CV RMSE, ciclo Δ→r→Δ)
    C3: Compatibilidad espectral con diagnóstico causal
        C3a (decoder): evalúa modelo directo Δ→r
        C3b (cycle): evalúa ciclo completo r→Δ̂→r̂
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    resolve_spectrum_path as resolve_run_spectrum_path,
    write_manifest,
    write_stage_summary,
    sha256_file,
)
# Silenciar warnings de convergencia en CV con pocos datos
warnings.filterwarnings("ignore", category=UserWarning)

__version__ = "1.5.0"


# =============================================================================
# Configuración
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Configuración del Bloque C."""
    run: str
    test_mode: bool = False  # Ejecuta self-tests deterministas y sale
    spectrum_file: str = "outputs/spectrum.h5"
    exp04_compare_run: Optional[str] = None
    export_atlas_points: bool = False
    
    # Features
    k_features: int = 3              # Número de ratios r_n (n=1..K)
    
    # Modelos a evaluar (inverso: r→Δ)
    models: tuple = ("linear", "poly2")
    
    # Modelo directo (Δ→r) para C3
    direct_model: str = "poly"       # "linear" o "poly"
    direct_degree: int = 4           # Grado del polinomio directo
    
    # Validación
    cv_folds: int = 5
    n_bootstrap: int = 200
    random_seed: int = 42
    
    # Contratos C1
    tau_delta: float = 0.02          # Tolerancia absoluta para C1
    sigma_factor: float = 2.0        # Factor para criterio 2σ
    sigma_cap: float = 0.1           # Techo para σ usable en C1
    
    # Contrato C3
    enable_c3: bool = False          # Activar C3 completo
    c3_metric: str = "rmse"          # rmse, rmse_log, rmse_rel
    c3_threshold: float = 0.05       # Umbral base para C3
    c3_weights: str = "none"         # none, inv_n, inv_n2, inv_n4, inv_r2
    
    # C3 Noise floor adaptativo
    c3_noise_floor_eps: float = 0.001
    c3_noise_floor_metric: Optional[str] = None  # None/"same" = igual a c3_metric
    c3_adaptive_threshold: bool = False
    c3_threshold_factor: float = 5.0
    
    # C3 Oracle test
    c3_oracle: bool = False
    
    # Targets Ising 3D (bootstrap precision, arXiv:2411.15300)
    delta_sigma: float = 0.518148806
    delta_epsilon: float = 1.41262528


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Bloque C: Diccionario holográfico inverso (v1.5.0)"
    )
    p.add_argument("--run", type=str, default=None, help="Nombre del run")
    p.add_argument("--test", action="store_true", dest="test_mode",
                   help="Ejecuta self-tests deterministas (T1/T2) y sale")
    p.add_argument("--spectrum-file", type=str, default="outputs/spectrum.h5",
                   dest="spectrum_file", help="Archivo H5 del espectro")
    p.add_argument(
        "--export-atlas-points",
        action="store_true",
        dest="export_atlas_points",
        help="Exporta atlas_points.json (opcional, default: desactivado)",
    )
    p.add_argument("--k-features", type=int, default=3, dest="k_features",
                   help="Número de ratios r_n a usar como features (default: 3)")
    p.add_argument("--exp04-compare-run", type=str, default=None,
                   dest="exp04_compare_run",
                   help="Run baseline para comparar contratos C5 (Exp04)")
    p.add_argument("--cv-folds", type=int, default=5, dest="cv_folds")
    p.add_argument("--n-bootstrap", type=int, default=200, dest="n_bootstrap")
    p.add_argument("--seed", type=int, default=42, dest="random_seed")
    p.add_argument("--tau-delta", type=float, default=0.02, dest="tau_delta")
    p.add_argument("--sigma-cap", type=float, default=0.1, dest="sigma_cap",
                   help="Techo máximo de σ para WEAK_PASS en C1 (default: 0.1)")
    
    # C3 básico
    p.add_argument("--enable-c3", action="store_true", dest="enable_c3",
                   help="Activar C3 completo con modelo directo Δ→ratios")
    p.add_argument("--direct-model", type=str, default="poly", dest="direct_model",
                   choices=["linear", "poly"], help="Tipo de modelo directo (default: poly)")
    p.add_argument("--direct-degree", type=int, default=4, dest="direct_degree",
                   help="Grado del polinomio para modelo directo (default: 4)")
    p.add_argument("--c3-metric", type=str, default="rmse", dest="c3_metric",
                   choices=["rmse", "rmse_log", "rmse_rel"],
                   help="Métrica para C3 (default: rmse)")
    p.add_argument("--c3-threshold", type=float, default=0.05, dest="c3_threshold",
                   help="Umbral base para C3 (default: 0.05)")
    
    # C3 pesos
    p.add_argument("--c3-weights", type=str, default="none", dest="c3_weights",
                   choices=["none", "inv_n", "inv_n2", "inv_n4", "inv_r2"],
                   help="Esquema de pesos para C3 (default: none)")
    
    # C3 noise floor adaptativo
    p.add_argument("--c3-noise-floor-eps", type=float, default=0.001,
                   dest="c3_noise_floor_eps",
                   help="Epsilon para cálculo de noise floor (default: 0.001)")
    p.add_argument("--c3-noise-floor-metric", type=str, default="same",
                   dest="c3_noise_floor_metric",
                   choices=["same", "rmse", "rmse_log", "rmse_rel"],
                   help="Métrica para noise floor (default: same = igual a --c3-metric)")
    p.add_argument("--c3-adaptive-threshold", action="store_true",
                   dest="c3_adaptive_threshold",
                   help="Usar umbral adaptativo basado en noise floor")
    p.add_argument("--c3-threshold-factor", type=float, default=5.0,
                   dest="c3_threshold_factor",
                   help="Factor multiplicador para umbral adaptativo (default: 5.0)")
    
    # C3 oracle
    p.add_argument("--c3-oracle", action="store_true", dest="c3_oracle",
                   help="Activar oracle test (comparación forward real vs aproximado)")
    
    args = p.parse_args()
    if not getattr(args, "test_mode", False) and args.run is None:
        p.error("--run es requerido salvo cuando se usa --test")
    return Config(
        run=(args.run if args.run is not None else ("__selftest__" if getattr(args, "test_mode", False) else None)),
        test_mode=getattr(args, "test_mode", False),
        spectrum_file=args.spectrum_file,
        export_atlas_points=args.export_atlas_points,
        k_features=args.k_features,
        exp04_compare_run=args.exp04_compare_run,
        cv_folds=args.cv_folds,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        tau_delta=args.tau_delta,
        sigma_cap=args.sigma_cap,
        enable_c3=args.enable_c3,
        direct_model=args.direct_model,
        direct_degree=args.direct_degree,
        c3_metric=args.c3_metric,
        c3_threshold=args.c3_threshold,
        c3_weights=args.c3_weights,
        c3_noise_floor_eps=args.c3_noise_floor_eps,
        c3_noise_floor_metric=args.c3_noise_floor_metric,
        c3_adaptive_threshold=args.c3_adaptive_threshold,
        c3_threshold_factor=args.c3_threshold_factor,
        c3_oracle=args.c3_oracle,
    )


# =============================================================================
# Carga de datos
# =============================================================================

def load_spectrum(h5_path: Path) -> dict:
    """Carga espectro desde H5 (output del Bloque B)."""
    try:
        import h5py
    except ImportError:
        print("ERROR: pip install h5py", file=sys.stderr)
        sys.exit(1)
    
    with h5py.File(h5_path, "r") as h5:
        dual_spectrum = bool(h5.attrs.get("dual_spectrum", False))
        data = {
            "delta_uv": h5["delta_uv"][:],
            "m2L2": h5["m2L2"][:],
            "z_grid": h5["z_grid"][:],
            "d": int(h5.attrs["d"]),
            "L": float(h5.attrs["L"]),
            "n_delta": int(h5.attrs["n_delta"]),
            "n_modes": int(h5.attrs["n_modes"]),
            "dual_spectrum": dual_spectrum,
        }
        if dual_spectrum:
            data["M2_D"] = h5["M2_D"][:]
            data["M2_N"] = h5["M2_N"][:]
            data["M2"] = h5["M2"][:] if "M2" in h5 else data["M2_D"]
        else:
            data["M2"] = h5["M2"][:]
    return data


def resolve_input_spectrum_path(run: str, spectrum_file: str) -> Path:
    """Resuelve la ruta del espectro.

    Contrato IO (canónico):
      runs/<run>/spectrum/outputs/spectrum.h5

    Legado:
      runs/<run>/spectrum/spectrum.h5

    Si spectrum_file es ruta absoluta o empieza por 'runs/', se usa tal cual
    para evitar duplicación de prefijos.
    """
    sf = Path(spectrum_file)
    if sf.is_absolute() or spectrum_file.startswith("runs/"):
        return sf

    run_dir = get_run_dir(run)
    if spectrum_file in ("outputs/spectrum.h5", "spectrum.h5"):
        return resolve_run_spectrum_path(run_dir)

    stage_dir = run_dir / "spectrum"
    cand = stage_dir / spectrum_file
    if cand.exists():
        return cand

    return resolve_run_spectrum_path(run_dir)


def verify_invariants(X: np.ndarray, y: Optional[np.ndarray] = None, *, name: str = "X") -> None:
    """Asserts defensivos: shapes y finitud (NaN/Inf).

    Objetivo: evitar debugging silencioso cuando cambie la geometría o el solver.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} debe ser np.ndarray, recibido {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"{name} debe ser (n_samples, k), recibido {X.shape}")
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        raise ValueError(f"{name} contiene NaNs/Infs en indices {bad[:5].tolist()} (mostrando hasta 5).")

    if y is not None:
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y debe ser np.ndarray, recibido {type(y)}")
        if y.ndim != 1:
            raise ValueError(f"y debe ser (n_samples,), recibido {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch dimensional: {name}={X.shape}, y={y.shape}")
        if not np.isfinite(y).all():
            bad = np.argwhere(~np.isfinite(y))
            raise ValueError(f"y contiene NaNs/Infs en indices {bad[:5].tolist()} (mostrando hasta 5).")


def compute_ratio_features(M2: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """Calcula features invariantes de escala: r_n = M_n² / M_0².
    
    Args:
        M2: array (n_samples, n_modes) con autovalores
        k: número de ratios a calcular (n=1..k)
    
    Returns:
        X: array (n_samples, k) con ratios r_1, r_2, ..., r_k
    """
    n_samples, n_modes = M2.shape
    
    if k >= n_modes:
        raise ValueError(f"k={k} debe ser < n_modes={n_modes}")
    
    M0 = M2[:, 0:1]  # (n_samples, 1)
    
    # Evitar división por cero
    M0_safe = np.maximum(M0, eps)
    
    # Ratios r_1, r_2, ..., r_k
    X = M2[:, 1:k+1] / M0_safe

    verify_invariants(X, name="X_ratios")

    return X


def build_features_from_spectrum(spectrum: dict, k: int, eps: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Construye features X desde el espectro, soportando dual spectrum."""
    dual = bool(spectrum.get("dual_spectrum", False))
    if dual:
        rD = compute_ratio_features(spectrum["M2_D"], k, eps)
        rN = compute_ratio_features(spectrum["M2_N"], k, eps)
        X = np.concatenate([rD, rN], axis=1)
        dim = 2 * k
    else:
        X = compute_ratio_features(spectrum["M2"], k, eps)
        dim = k

    features_meta = {
        "k": int(k),
        "dual": dual,
        "dim": int(dim),
        "definition": "ratios M2_n/M2_0 excluding n=0",
    }
    return X, features_meta


# =============================================================================
# Modelos paramétricos (inverso: r → Δ)
# =============================================================================

def build_design_matrix(X: np.ndarray, model_type: str) -> tuple[np.ndarray, list]:
    """Construye matriz de diseño para el modelo especificado.
    
    Args:
        X: array (n_samples, k_features) con ratios
        model_type: "linear" o "poly2"
    
    Returns:
        Phi: matriz de diseño (n_samples, n_params)
        feature_names: lista de nombres de features
    """
    n_samples, k = X.shape
    
    if model_type == "linear":
        # [1, r_1, r_2, ..., r_k]
        Phi = np.hstack([np.ones((n_samples, 1)), X])
        names = ["1"] + [f"r_{i+1}" for i in range(k)]
        
    elif model_type == "poly2":
        # [1, r_i, r_i², r_i*r_j]
        features = [np.ones((n_samples, 1))]
        names = ["1"]
        
        # Términos lineales
        for i in range(k):
            features.append(X[:, i:i+1])
            names.append(f"r_{i+1}")
        
        # Términos cuadráticos
        for i in range(k):
            features.append(X[:, i:i+1] ** 2)
            names.append(f"r_{i+1}^2")
        
        # Términos cruzados
        for i in range(k):
            for j in range(i+1, k):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                names.append(f"r_{i+1}*r_{j+1}")
        
        Phi = np.hstack(features)
    
    else:
        raise ValueError(f"model_type desconocido: {model_type}")
    
    return Phi, names


def fit_model(Phi: np.ndarray, y: np.ndarray, ridge_lambda: float = 1e-8) -> np.ndarray:
    """Ajusta modelo por mínimos cuadrados (con ridge mínimo para estabilidad).

    Returns:
        beta: coeficientes (n_params,)
    """

    verify_invariants(Phi, y, name="Phi")
    n_params = Phi.shape[1]
    
    # Normal equations con ridge: (Φᵀ Φ + λI)⁻¹ Φᵀ y
    A = Phi.T @ Phi + ridge_lambda * np.eye(n_params)
    b = Phi.T @ y
    
    beta = np.linalg.solve(A, b)
    return beta


def predict(Phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predicción: ŷ = Φ β"""
    return Phi @ beta


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de ajuste."""
    residuals = y_true - y_pred
    n = len(y_true)
    
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "ss_res": float(ss_res),
        "n": n,
    }


def compute_information_criteria(y: np.ndarray, y_pred: np.ndarray, n_params: int) -> dict:
    """Calcula AIC y BIC."""
    n = len(y)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    
    sigma2_mle = ss_res / n
    log_likelihood = -n/2 * (np.log(2 * np.pi) + np.log(sigma2_mle + 1e-10) + 1)
    
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n)
    
    return {
        "aic": float(aic),
        "bic": float(bic),
        "log_likelihood": float(log_likelihood),
        "n_params": n_params,
    }


# =============================================================================
# Validación cruzada
# =============================================================================

def cross_validate(X: np.ndarray, y: np.ndarray, model_type: str, 
                   n_folds: int, seed: int) -> dict:
    """K-fold cross validation."""
    np.random.seed(seed)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    
    fold_size = n_samples // n_folds
    cv_errors = []
    cv_predictions = np.zeros(n_samples)
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        Phi_train, _ = build_design_matrix(X_train, model_type)
        beta = fit_model(Phi_train, y_train)
        
        Phi_test, _ = build_design_matrix(X_test, model_type)
        y_pred = predict(Phi_test, beta)
        
        cv_predictions[test_idx] = y_pred
        fold_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        cv_errors.append(fold_rmse)
    
    return {
        "cv_rmse_mean": float(np.mean(cv_errors)),
        "cv_rmse_std": float(np.std(cv_errors)),
        "cv_rmse_per_fold": [float(e) for e in cv_errors],
        "cv_predictions": cv_predictions,
    }


# =============================================================================
# Bootstrap para incertidumbre
# =============================================================================

def bootstrap_uncertainty(X: np.ndarray, y: np.ndarray, model_type: str,
                          n_bootstrap: int, seed: int) -> dict:
    """Estima incertidumbre en predicciones via bootstrap."""
    np.random.seed(seed)
    n_samples = len(y)
    
    Phi_full, _ = build_design_matrix(X, model_type)
    beta_full = fit_model(Phi_full, y)
    y_pred_full = predict(Phi_full, beta_full)
    
    bootstrap_predictions = np.zeros((n_bootstrap, n_samples))
    bootstrap_betas = []
    
    for b in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_b, y_b = X[idx], y[idx]
        
        Phi_b, _ = build_design_matrix(X_b, model_type)
        beta_b = fit_model(Phi_b, y_b)
        bootstrap_betas.append(beta_b)
        
        y_pred_b = predict(Phi_full, beta_b)
        bootstrap_predictions[b] = y_pred_b
    
    sigma_delta = np.std(bootstrap_predictions, axis=0)
    sigma_delta_p90 = float(np.percentile(sigma_delta, 90))
    beta_mean = np.mean(bootstrap_betas, axis=0)
    beta_std = np.std(bootstrap_betas, axis=0)
    
    return {
        "sigma_delta": sigma_delta,
        "sigma_delta_mean": float(np.mean(sigma_delta)),
        "sigma_delta_p90": sigma_delta_p90,
        "sigma_delta_max": float(np.max(sigma_delta)),
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "n_bootstrap": n_bootstrap,
    }


# =============================================================================
# Selección de modelo
# =============================================================================

def select_best_model(X: np.ndarray, y: np.ndarray, models: tuple,
                      cv_folds: int, seed: int) -> dict:
    """Evalúa modelos y selecciona el mejor por BIC."""
    results = {}
    
    for model_type in models:
        Phi, feature_names = build_design_matrix(X, model_type)
        beta = fit_model(Phi, y)
        y_pred = predict(Phi, beta)
        
        metrics = compute_metrics(y, y_pred)
        info_criteria = compute_information_criteria(y, y_pred, len(beta))
        cv_results = cross_validate(X, y, model_type, cv_folds, seed)
        
        results[model_type] = {
            "beta": beta,
            "feature_names": feature_names,
            "n_params": len(beta),
            "metrics": metrics,
            "info_criteria": info_criteria,
            "cv": cv_results,
        }
    
    best_model = min(results.keys(), key=lambda m: results[m]["info_criteria"]["bic"])
    
    return {
        "all_models": results,
        "best_model": best_model,
        "selection_criterion": "BIC",
    }


# =============================================================================
# Contratos C1 y C2
# =============================================================================

def evaluate_c1_ising(delta_pred: np.ndarray, sigma_delta: np.ndarray,
                      delta_target: float, tau: float, sigma_factor: float,
                      target_name: str, sigma_cap: float = 0.1,
                      delta_range: tuple[float, float] = None) -> dict:
    """Contrato C1: compatibilidad puntual con Ising."""
    errors = np.abs(delta_pred - delta_target)
    
    in_domain = True
    if delta_range is not None:
        delta_min, delta_max = delta_range
        in_domain = delta_min <= delta_target <= delta_max
    
    if not in_domain:
        best_idx = np.argmin(errors)
        return {
            "target_name": target_name,
            "target_value": delta_target,
            "tolerance_tau": tau,
            "sigma_factor": sigma_factor,
            "sigma_cap": sigma_cap,
            "delta_range": list(delta_range) if delta_range else None,
            "in_domain": False,
            "best_candidate": {
                "index": int(best_idx),
                "delta_pred": float(delta_pred[best_idx]),
                "sigma_delta": float(sigma_delta[best_idx]),
                "error": float(errors[best_idx]),
                "status": "OUT_OF_DOMAIN",
                "note": f"Target Δ={delta_target:.4f} fuera del rango [{delta_min:.3f}, {delta_max:.3f}]",
            },
            "global_status": "OUT_OF_DOMAIN",
            "n_pass_tau": 0,
            "n_pass_sigma": 0,
            "n_sigma_too_large": 0,
        }
    
    pass_tau = errors <= tau
    sigma_usable = sigma_delta <= sigma_cap
    pass_sigma_raw = errors <= sigma_factor * sigma_delta
    pass_sigma = pass_sigma_raw & sigma_usable
    
    scores = np.where(pass_tau, 0, np.where(pass_sigma, 1, 2))
    combined_score = scores + errors / (errors.max() + 1e-10)
    best_idx = np.argmin(combined_score)
    
    if pass_tau[best_idx]:
        status = "STRONG_PASS"
    elif pass_sigma[best_idx]:
        status = "WEAK_PASS"
    elif pass_sigma_raw[best_idx] and not sigma_usable[best_idx]:
        status = "FAIL_SIGMA_TOO_LARGE"
    else:
        status = "FAIL"
    
    if np.any(pass_tau):
        global_status = "STRONG_PASS"
    elif np.any(pass_sigma):
        global_status = "WEAK_PASS"
    else:
        global_status = "FAIL"
    
    return {
        "target_name": target_name,
        "target_value": delta_target,
        "tolerance_tau": tau,
        "sigma_factor": sigma_factor,
        "sigma_cap": sigma_cap,
        "delta_range": list(delta_range) if delta_range else None,
        "in_domain": True,
        "best_candidate": {
            "index": int(best_idx),
            "delta_pred": float(delta_pred[best_idx]),
            "sigma_delta": float(sigma_delta[best_idx]),
            "error": float(errors[best_idx]),
            "status": status,
            "pass_tau": bool(pass_tau[best_idx]),
            "pass_sigma": bool(pass_sigma[best_idx]),
            "sigma_usable": bool(sigma_usable[best_idx]),
        },
        "global_status": global_status,
        "n_pass_tau": int(np.sum(pass_tau)),
        "n_pass_sigma": int(np.sum(pass_sigma)),
        "n_sigma_too_large": int(np.sum(pass_sigma_raw & ~sigma_usable)),
    }


def evaluate_c2_consistency(y_true: np.ndarray, y_pred: np.ndarray,
                            cv_rmse: float) -> dict:
    """Contrato C2: consistencia interna del diccionario."""
    metrics = compute_metrics(y_true, y_pred)
    consistency_ok = cv_rmse < 0.05
    
    return {
        "fit_rmse": metrics["rmse"],
        "fit_mae": metrics["mae"],
        "fit_r2": metrics["r2"],
        "cv_rmse": cv_rmse,
        "threshold_cv_rmse": 0.05,
        "consistency_ok": consistency_ok,
    }


# =============================================================================
# Contrato C3: Modelo directo y métricas
# =============================================================================

# =============================================================================
# Modelo Directo (Proxy para C3)
# NOTA: Este modelo polinomial NO es la física real. Es un proxy suave local
# usado para evaluar la invertibilidad del mapeo.
# Si C3a falla, significa que el proxy es malo, no que la física esté rota.
# =============================================================================


def build_direct_design_matrix(y: np.ndarray, degree: int) -> np.ndarray:
    """Construye matriz de diseño polinomial para modelo directo Δ→r."""
    n_samples = len(y)
    # [1, Δ, Δ², ..., Δ^degree]
    Phi = np.column_stack([y**p for p in range(degree + 1)])
    return Phi


def fit_direct_model(y: np.ndarray, X: np.ndarray, degree: int) -> list[np.ndarray]:
    """Ajusta modelo directo Δ → ratios.
    
    Args:
        y: array (n_samples,) con Δ
        X: array (n_samples, k) con ratios objetivo
        degree: grado del polinomio
    
    Returns:
        betas: lista de coeficientes, uno por ratio
    """
    n_samples, k = X.shape
    Phi = build_direct_design_matrix(y, degree)
    
    betas = []
    for j in range(k):
        beta_j = fit_model(Phi, X[:, j])
        betas.append(beta_j)
    
    return betas


def predict_ratios(y: np.ndarray, betas: list[np.ndarray], degree: int) -> np.ndarray:
    """Predice ratios desde Δ usando modelo directo."""
    n_samples = len(y)
    k = len(betas)
    Phi = build_direct_design_matrix(y, degree)
    
    X_pred = np.zeros((n_samples, k))
    for j, beta_j in enumerate(betas):
        X_pred[:, j] = Phi @ beta_j
    
    return X_pred


def compute_weights(k: int, scheme: str, X: np.ndarray = None) -> np.ndarray:
    """Calcula pesos para métricas C3.
    
    Args:
        k: número de ratios
        scheme: "none", "inv_n", "inv_n2", "inv_n4", "inv_r2"
        X: array (n_samples, k) de ratios (requerido para inv_r2)
    
    Returns:
        weights: array (k,) con pesos normalizados
    """
    n = np.arange(1, k + 1)  # n = 1, 2, ..., k
    
    if scheme == "none":
        w = np.ones(k)
    elif scheme == "inv_n":
        w = 1.0 / n
    elif scheme == "inv_n2":
        w = 1.0 / (n ** 2)
    elif scheme == "inv_n4":
        w = 1.0 / (n ** 4)
    elif scheme == "inv_r2":
        if X is None:
            raise ValueError("inv_r2 requiere X (ratios)")
        # Media de r_n² sobre muestras + epsilon para estabilidad
        r_mean_sq = np.mean(X ** 2, axis=0) + 1e-12
        w = 1.0 / r_mean_sq
    else:
        raise ValueError(f"Esquema de pesos desconocido: {scheme}")
    
    # Normalizar para que sumen 1
    w = w / np.sum(w)
    return w


def compute_ratio_distance(
    r_true: np.ndarray, 
    r_pred: np.ndarray, 
    metric: str,
    weights: np.ndarray = None
) -> dict:
    """Función unificada para calcular distancia entre ratios.
    
    Args:
        r_true: array (n_samples, k) ratios verdaderos
        r_pred: array (n_samples, k) ratios predichos
        metric: "rmse", "rmse_log", "rmse_rel"
        weights: array (k,) pesos por ratio (opcional)
    
    Returns:
        dict con global, per_ratio, weights_used
    """
    n_samples, k = r_true.shape
    
    if weights is None:
        weights = np.ones(k) / k  # Pesos uniformes
    
    # Calcular errores según métrica
    if metric == "rmse":
        errors = r_true - r_pred  # (n_samples, k)
        errors_sq = errors ** 2
    elif metric == "rmse_log":
        # log(r) con clamp para evitar log(0)
        r_true_safe = np.maximum(r_true, 1e-10)
        r_pred_safe = np.maximum(r_pred, 1e-10)
        errors = np.log(r_true_safe) - np.log(r_pred_safe)
        errors_sq = errors ** 2
    elif metric == "rmse_rel":
        # Error relativo: (r - r̂) / r
        r_true_safe = np.where(np.abs(r_true) > 1e-10, r_true, 1e-10)
        errors = (r_true - r_pred) / r_true_safe
        errors_sq = errors ** 2
    else:
        raise ValueError(f"Métrica desconocida: {metric}")
    
    # MSE por ratio (promedio sobre muestras)
    mse_per_ratio = np.mean(errors_sq, axis=0)  # (k,)
    rmse_per_ratio = np.sqrt(mse_per_ratio)
    
    # WRMSE global: sqrt(sum_n w_n * mse_n)
    wrmse_global = np.sqrt(np.sum(weights * mse_per_ratio))
    
    return {
        "global": float(wrmse_global),
        "per_ratio": [float(r) for r in rmse_per_ratio],
        "weights_used": [float(w) for w in weights],
        "metric": metric,
    }


def estimate_sensitivity(delta: np.ndarray, ratios: np.ndarray) -> dict:
    """Estima sensibilidad ||dr/dΔ||_2 usando diferencias finitas en datos."""
    if delta.ndim != 1:
        raise ValueError("delta debe ser 1D")
    if ratios.ndim != 2:
        raise ValueError("ratios debe ser 2D")
    if len(delta) != ratios.shape[0]:
        raise ValueError("delta y ratios deben tener mismo n_samples")

    n_samples = len(delta)
    if n_samples < 2:
        return {
            "n_samples": n_samples,
            "s_p50": 0.0,
            "s_p90": 0.0,
            "s_p95": 0.0,
            "s_max": 0.0,
        }

    order = np.argsort(delta)
    delta_sorted = delta[order]
    ratios_sorted = ratios[order]

    sensitivities = np.zeros(n_samples)
    for i in range(n_samples):
        if i == 0:
            dr = ratios_sorted[1] - ratios_sorted[0]
            ddelta = delta_sorted[1] - delta_sorted[0]
        elif i == n_samples - 1:
            dr = ratios_sorted[-1] - ratios_sorted[-2]
            ddelta = delta_sorted[-1] - delta_sorted[-2]
        else:
            dr = ratios_sorted[i + 1] - ratios_sorted[i - 1]
            ddelta = delta_sorted[i + 1] - delta_sorted[i - 1]

        denom = ddelta if np.abs(ddelta) > 1e-12 else 1e-12
        sensitivities[i] = np.linalg.norm(dr / denom)

    return {
        "n_samples": n_samples,
        "s_p50": float(np.percentile(sensitivities, 50)),
        "s_p90": float(np.percentile(sensitivities, 90)),
        "s_p95": float(np.percentile(sensitivities, 95)),
        "s_max": float(np.max(sensitivities)),
    }


def compute_sensitivity(
    y: np.ndarray,
    betas_direct: list[np.ndarray],
    degree: int,
    eps: float = 0.001
) -> dict:
    """Calcula sensibilidad del modelo directo: |r(Δ+ε) - r(Δ)| / ε.
    
    Proxy de condicionamiento del observable.
    """
    n_samples = len(y)
    k = len(betas_direct)
    
    # r(Δ) y r(Δ+ε)
    r_base = predict_ratios(y, betas_direct, degree)
    r_perturbed = predict_ratios(y + eps, betas_direct, degree)
    
    # Sensibilidad por componente: |dr/dΔ| ≈ |r(Δ+ε) - r(Δ)| / ε
    sensitivity = np.abs(r_perturbed - r_base) / eps  # (n_samples, k)
    
    # Agregar estadísticas
    sensitivity_median_per_ratio = np.median(sensitivity, axis=0)  # (k,)
    
    # Norma L2 de sensibilidad por muestra, luego mediana
    sensitivity_norm_per_sample = np.linalg.norm(sensitivity, axis=1)  # (n_samples,)
    sensitivity_norm_median = float(np.median(sensitivity_norm_per_sample))
    
    return {
        "eps": eps,
        "per_ratio_median": [float(s) for s in sensitivity_median_per_ratio],
        "norm_median": sensitivity_norm_median,
        "norm_max": float(np.max(sensitivity_norm_per_sample)),
        "norm_p90": float(np.percentile(sensitivity_norm_per_sample, 90)),
    }


def compute_noise_floor(
    y: np.ndarray,
    X: np.ndarray,
    betas_direct: list[np.ndarray],
    degree: int,
    eps: float,
    metric: str,
    weights: np.ndarray
) -> dict:
    """Estima noise floor: resolución efectiva del observable bajo perturbación.
    
    Mide cuánto cambia r cuando Δ cambia por ±ε.
    """
    n_samples = len(y)
    
    # r predicho en Δ y en Δ+ε
    r_base = predict_ratios(y, betas_direct, degree)
    r_perturbed = predict_ratios(y + eps, betas_direct, degree)
    
    # Calcular "error" de la perturbación usando la métrica elegida
    distance = compute_ratio_distance(r_base, r_perturbed, metric, weights)
    
    # También calcular por muestra para estadísticas
    sigmas = []
    for i in range(n_samples):
        sample_dist = compute_ratio_distance(
            r_base[i:i+1], r_perturbed[i:i+1], metric, weights
        )
        sigmas.append(sample_dist["global"])
    
    sigmas = np.array(sigmas)
    
    return {
        "eps": eps,
        "metric": metric,
        "median_sigma": float(np.median(sigmas)),
        "mean_sigma": float(np.mean(sigmas)),
        "p90_sigma": float(np.percentile(sigmas, 90)),
        "max_sigma": float(np.max(sigmas)),
        "aggregate_distance": distance["global"],
    }


def evaluate_c3a_decoder(
    X_true: np.ndarray,
    y_true: np.ndarray,
    betas_direct: list[np.ndarray],
    degree: int,
    metric: str,
    weights: np.ndarray
) -> dict:
    """C3a: Evalúa modelo directo (decoder) Δ_true → r̂.
    
    Compara r_true con r̂ = decoder(Δ_true).
    """
    r_pred = predict_ratios(y_true, betas_direct, degree)
    
    # Calcular todas las métricas
    result = {}
    for m in ["rmse", "rmse_log", "rmse_rel"]:
        dist = compute_ratio_distance(X_true, r_pred, m, weights if m == metric else None)
        result[m] = dist["global"]
        if m == metric:
            result["per_ratio"] = dist["per_ratio"]
    
    result["metric_primary"] = metric
    result["global"] = result[metric]
    
    return result


def evaluate_c3b_cycle(
    X_true: np.ndarray,
    y_true: np.ndarray,
    inverse_model_type: str,
    betas_direct: list[np.ndarray],
    degree: int,
    metric: str,
    weights: np.ndarray
) -> dict:
    """C3b: Evalúa ciclo completo r → Δ̂ → r̂.
    
    1. Ajusta modelo inverso r → Δ
    2. Predice Δ̂ desde r_true
    3. Predice r̂ = decoder(Δ̂)
    4. Compara r_true vs r̂
    """
    # Modelo inverso
    Phi_inv, _ = build_design_matrix(X_true, inverse_model_type)
    beta_inv = fit_model(Phi_inv, y_true)
    delta_pred = predict(Phi_inv, beta_inv)
    
    # Ciclo: r → Δ̂ → r̂
    r_reconstructed = predict_ratios(delta_pred, betas_direct, degree)
    
    # Calcular todas las métricas
    result = {}
    for m in ["rmse", "rmse_log", "rmse_rel"]:
        dist = compute_ratio_distance(X_true, r_reconstructed, m, weights if m == metric else None)
        result[m] = dist["global"]
        if m == metric:
            result["per_ratio"] = dist["per_ratio"]
    
    result["metric_primary"] = metric
    result["global"] = result[metric]
    result["delta_pred_range"] = [float(delta_pred.min()), float(delta_pred.max())]
    
    return result


def evaluate_c3_oracle(
    X_true: np.ndarray,
    y_true: np.ndarray,
    betas_direct: list[np.ndarray],
    degree: int,
    metric: str,
    weights: np.ndarray
) -> dict:
    """Oracle test: compara forward "real" (datos) vs forward aproximado (decoder).
    
    En baseline, esto es equivalente a C3a. Pero se etiqueta explícitamente
    para hacer el gap entre forward real y aproximado defendible.
    """
    # En baseline, el "forward real" son los ratios del dataset
    # El "forward aproximado" es el decoder
    # Esto es idéntico a C3a, pero con etiqueta explícita
    
    r_pred = predict_ratios(y_true, betas_direct, degree)
    
    result = {}
    for m in ["rmse", "rmse_log", "rmse_rel"]:
        dist = compute_ratio_distance(X_true, r_pred, m, weights if m == metric else None)
        result[m] = dist["global"]
        if m == metric:
            result["per_ratio"] = dist["per_ratio"]
    
    result["metric_primary"] = metric
    result["global"] = result[metric]
    result["description"] = "Gap entre forward real (datos) y forward aproximado (decoder)"
    
    return result


def evaluate_c3_full(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    inverse_model_type: str,
    sigma_delta_stats: Optional[dict] = None
) -> dict:
    """Evaluación completa de C3 con diagnóstico causal."""

    verify_invariants(X, y, name="X")

    if not cfg.enable_c3:
        return {
            "status": "SKIP",
            "failure_mode": None,
            "note": "C3 requiere --enable-c3. Usar --enable-c3 para activar.",
        }
    
    k = X.shape[1]
    metric = cfg.c3_metric
    noise_floor_metric = metric
    if cfg.c3_noise_floor_metric not in (None, "same"):
        noise_floor_metric = cfg.c3_noise_floor_metric
    
    # Invariantes defensivos
    verify_invariants(X, y, name="X_ratios")

    # Calcular pesos
    weights = compute_weights(k, cfg.c3_weights, X)
    
    # Ajustar modelo directo
    betas_direct = fit_direct_model(y, X, cfg.direct_degree)
    
    # C3a: Decoder
    c3a = evaluate_c3a_decoder(X, y, betas_direct, cfg.direct_degree, metric, weights)
    
    # C3b: Cycle
    c3b = evaluate_c3b_cycle(X, y, inverse_model_type, betas_direct, cfg.direct_degree, metric, weights)
    
    # Sensitivity (datos reales)
    sensitivity = estimate_sensitivity(y, X)
    
    # Noise floor
    noise_floor = compute_noise_floor(
        y, X, betas_direct, cfg.direct_degree,
        cfg.c3_noise_floor_eps, noise_floor_metric, weights
    )
    
    # Oracle (opcional)
    oracle = None
    if cfg.c3_oracle:
        oracle = evaluate_c3_oracle(X, y, betas_direct, cfg.direct_degree, metric, weights)
    
    # Calcular umbral efectivo (C3b)
    threshold_user = cfg.c3_threshold
    noise_floor_r = noise_floor["median_sigma"]
    if sigma_delta_stats is None:
        sigma_delta_used = 0.0
        sigma_delta_source = "none"
    else:
        if "sigma_delta_p90" in sigma_delta_stats:
            sigma_delta_used = float(sigma_delta_stats["sigma_delta_p90"])
            sigma_delta_source = "p90"
        else:
            sigma_delta_used = float(sigma_delta_stats.get("sigma_delta_mean", 0.0))
            sigma_delta_source = "mean"

    tol_cycle = max(noise_floor_r, sensitivity["s_p90"] * sigma_delta_used)
    threshold_used = cfg.c3_threshold_factor * tol_cycle if cfg.c3_adaptive_threshold else threshold_user
    threshold_mode = "adaptive" if cfg.c3_adaptive_threshold else "user"
    
    # Determinar failure_mode y status
    c3a_value = c3a["global"]
    c3b_value = c3b["global"]

    c3a_ok = bool(c3a_value <= threshold_user)
    c3b_ok = bool(c3b_value <= threshold_used)
    c3a_status = "PASS" if c3a_ok else "FAIL"
    c3b_status = "PASS" if c3b_ok else "FAIL"
    
    if c3a_value > threshold_user:
        failure_mode = "DECODER_MISMATCH"
        status = "FAIL"
    elif c3b_value > threshold_used:
        failure_mode = "CYCLE_INCONSISTENT"
        status = "FAIL"
    else:
        failure_mode = None
        status = "PASS"
    
    return {
        "status": status,
        "c3a_status": c3a_status,
        "c3b_status": c3b_status,
        "failure_mode": failure_mode,
        "metric": metric,
        "weights": {
            "scheme": cfg.c3_weights,
            "values": [float(w) for w in weights],
        },
        "c3a_decoder": c3a,
        "c3b_cycle": c3b,
        "sensitivity": sensitivity,
        "sigma_delta": {
            "used": float(sigma_delta_used),
            "source": sigma_delta_source,
        },
        "tol_cycle": float(tol_cycle),
        "noise_floor": noise_floor,
        "threshold": {
            "user": threshold_user,
            "adaptive": float(cfg.c3_threshold_factor * tol_cycle),
            "effective": float(threshold_used),
            "factor": cfg.c3_threshold_factor,
            "mode": threshold_mode,
        },
        "threshold_used": float(threshold_used),
        "oracle": oracle,
        "direct_model": {
            "type": cfg.direct_model,
            "degree": cfg.direct_degree,
        },
        # Backward compatibility
        "spectral_ok": status == "PASS",
        "rmse_global": c3b_value,  # Para compatibilidad con código anterior
    }


# =============================================================================
# Atlas de teorías efectivas
# =============================================================================

def build_atlas(delta: np.ndarray, M2: np.ndarray, k: int) -> dict:
    """Construye atlas simple de teorías efectivas."""
    X = compute_ratio_features(M2, k)
    n_theories = len(delta)
    
    theories = []
    for i in range(n_theories):
        theories.append({
            "id": i,
            "delta": float(delta[i]),
            "M2_0": float(M2[i, 0]),
            "ratios": X[i].tolist(),
            "regime": "UV" if delta[i] < 2.0 else ("IR" if delta[i] > 4.0 else "intermediate"),
        })
    
    return {
        "n_theories": n_theories,
        "delta_range": [float(delta.min()), float(delta.max())],
        "theories": theories,
        "clustering_method": "parametric_sweep",
    }


# =============================================================================
# IO
# =============================================================================

def resolve_validation_path(run: str) -> Optional[Path]:
    """Resuelve validation.json para un run (outputs primero, luego legacy)."""
    run_dir = get_run_dir(run)
    candidates = [
        run_dir / "dictionary" / "outputs" / "validation.json",
        run_dir / "dictionary" / "validation.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_validation(run: str) -> tuple[Optional[dict], Optional[Path]]:
    """Carga validation.json de un run si existe."""
    path = resolve_validation_path(run)
    if path is None:
        return None, None
    with open(path, "r") as f:
        return json.load(f), path


def extract_primary_error(validation: dict) -> tuple[Optional[float], Optional[str]]:
    """Extrae la métrica de error principal según claves existentes."""
    c3 = validation.get("C3_spectral", {})
    c3b = c3.get("c3b_cycle", {})
    if isinstance(c3b, dict) and "global" in c3b:
        return float(c3b["global"]), "C3_spectral.c3b_cycle.global"
    c2 = validation.get("C2_consistency", {})
    if isinstance(c2, dict) and "cv_rmse" in c2:
        return float(c2["cv_rmse"]), "C2_consistency.cv_rmse"
    return None, None


def extract_c3b_status(validation: dict) -> Optional[str]:
    """Extrae status C3b si está disponible."""
    c3 = validation.get("C3_spectral", {})
    status = c3.get("c3b_status")
    if status in ("PASS", "FAIL"):
        return status
    return None


def evaluate_c5(
    current_validation: dict,
    *,
    dual: bool,
    compare_run: Optional[str],
    eps: float = 1e-12
) -> dict:
    """Evalúa contratos C5 (Exp04) comparando con baseline si procede."""
    if not dual:
        return {
            "status": "SKIP",
            "reason": "dual_spectrum False",
            "C5a": {"status": "SKIP", "reason": "dual_spectrum False"},
            "C5b": {"status": "SKIP", "reason": "dual_spectrum False"},
            "C5c": {"status": "SKIP", "reason": "dual_spectrum False"},
        }

    if not compare_run:
        return {
            "status": "SKIP",
            "reason": "no baseline provided",
            "C5a": {"status": "SKIP", "reason": "no baseline provided"},
            "C5b": {"status": "SKIP", "reason": "no baseline provided"},
            "C5c": {"status": "SKIP", "reason": "no baseline provided"},
        }

    baseline_validation, baseline_path = load_validation(compare_run)
    if baseline_validation is None:
        reason = f"baseline validation not found for {compare_run}"
        return {
            "status": "SKIP",
            "reason": reason,
            "baseline_run": compare_run,
            "C5a": {"status": "SKIP", "reason": reason},
            "C5b": {"status": "SKIP", "reason": reason},
            "C5c": {"status": "SKIP", "reason": reason},
        }

    c5a = {"status": "SKIP"}
    baseline_err, baseline_key = extract_primary_error(baseline_validation)
    dual_err, dual_key = extract_primary_error(current_validation)
    if baseline_err is None or dual_err is None:
        c5a["reason"] = "missing error metric in validation"
    else:
        gain = baseline_err - dual_err
        gain_frac = gain / max(baseline_err, eps)
        c5a = {
            "status": "PASS" if dual_err < baseline_err else "FAIL",
            "baseline_error": baseline_err,
            "dual_error": dual_err,
            "gain": gain,
            "gain_frac": gain_frac,
            "baseline_metric": baseline_key,
            "dual_metric": dual_key,
        }

    c5b = {"status": "SKIP"}
    baseline_c3b = extract_c3b_status(baseline_validation)
    dual_c3b = extract_c3b_status(current_validation)
    if baseline_c3b is None or dual_c3b is None:
        c5b["reason"] = "missing C3b status in validation"
    else:
        c5b = {
            "status": "PASS" if dual_c3b == "PASS" else "FAIL",
            "baseline_C3b": baseline_c3b,
            "dual_C3b": dual_c3b,
            "rule": "PASS if dual does not worsen; improves if baseline FAIL",
        }

    c5c = {"status": "SKIP"}
    baseline_c2 = baseline_validation.get("C2_consistency", {})
    dual_c2 = current_validation.get("C2_consistency", {})
    if "cv_rmse" not in baseline_c2 or "cv_rmse" not in dual_c2:
        c5c["reason"] = "missing cv_rmse in validation"
    else:
        baseline_rmse = float(baseline_c2["cv_rmse"])
        dual_rmse = float(dual_c2["cv_rmse"])
        ratio = dual_rmse / max(baseline_rmse, eps)
        c5c = {
            "status": "PASS" if dual_rmse <= 1.2 * baseline_rmse else "FAIL",
            "baseline_cv_rmse": baseline_rmse,
            "dual_cv_rmse": dual_rmse,
            "ratio": ratio,
            "threshold": 1.2,
        }

    return {
        "status": "EVALUATED",
        "baseline_run": compare_run,
        "baseline_validation_path": str(baseline_path),
        "C5a": c5a,
        "C5b": c5b,
        "C5c": c5c,
    }


def write_outputs(stage_dir: Path, outputs_dir: Path, cfg: Config, spectrum: dict,
                  model_selection: dict, bootstrap: dict,
                  contracts: dict, atlas: dict, features_meta: dict,
                  spectrum_rel_path: str, c5_contracts: dict) -> dict:
    """Escribe todos los outputs del Bloque C."""
    import h5py

    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    best_model = model_selection["best_model"]
    best_result = model_selection["all_models"][best_model]
    
    # --- dictionary.h5 ---
    h5_path = outputs_dir / "dictionary.h5"
    
    with h5py.File(h5_path, "w") as h5:
        feat_grp = h5.create_group("features")
        feat_grp.attrs["k"] = cfg.k_features
        feat_grp.attrs["definition"] = "r_n = M_n^2 / M_0^2, n=1..k"
        feat_grp.attrs["dual"] = bool(features_meta["dual"])
        feat_grp.attrs["dim"] = int(features_meta["dim"])
        feat_grp.create_dataset("feature_names", 
                                data=np.array(best_result["feature_names"], dtype="S"))
        
        model_grp = h5.create_group("model")
        model_grp.attrs["type"] = best_model
        model_grp.attrs["n_params"] = best_result["n_params"]
        model_grp.create_dataset("beta", data=best_result["beta"])
        model_grp.create_dataset("beta_bootstrap_mean", data=bootstrap["beta_mean"])
        model_grp.create_dataset("beta_bootstrap_std", data=bootstrap["beta_std"])
        
        sel_grp = h5.create_group("selection")
        sel_grp.attrs["criterion"] = "BIC"
        sel_grp.attrs["best_model"] = best_model
        for model_type, result in model_selection["all_models"].items():
            m_grp = sel_grp.create_group(model_type)
            m_grp.attrs["aic"] = result["info_criteria"]["aic"]
            m_grp.attrs["bic"] = result["info_criteria"]["bic"]
            m_grp.attrs["cv_rmse"] = result["cv"]["cv_rmse_mean"]
            m_grp.attrs["r2"] = result["metrics"]["r2"]
        
        unc_grp = h5.create_group("uncertainty")
        unc_grp.attrs["method"] = "bootstrap"
        unc_grp.attrs["n_bootstrap"] = bootstrap["n_bootstrap"]
        unc_grp.attrs["seed"] = cfg.random_seed
        unc_grp.create_dataset("sigma_delta", data=bootstrap["sigma_delta"])
        
        pred_grp = h5.create_group("predictions")
        pred_grp.create_dataset("delta_true", data=spectrum["delta_uv"])
        pred_grp.create_dataset("delta_pred", 
                                data=best_result["cv"]["cv_predictions"])
        
        h5.attrs["version"] = __version__
        h5.attrs["created"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["spectrum_source"] = cfg.spectrum_file
    
    # --- atlas.json ---
    atlas_path = outputs_dir / "atlas.json"
    with open(atlas_path, "w") as f:
        json.dump(atlas, f, indent=2)

    # --- atlas_points.json (opcional) ---
    atlas_points_path = outputs_dir / "atlas_points.json"
    atlas_points_feature_key = "ratios"
    atlas_points_generated = False
    atlas_points_status = "disabled"
    atlas_points_reason = "export disabled (--export-atlas-points not set)"
    exporter_path = Path("experiment/ringdown/export_atlas_points.py")
    if cfg.export_atlas_points:
        if exporter_path.exists():
            result = subprocess.run(
                [
                    sys.executable,
                    str(exporter_path),
                    "--atlas",
                    str(atlas_path),
                    "--out",
                    str(atlas_points_path),
                    "--feature-key",
                    atlas_points_feature_key,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and atlas_points_path.exists():
                atlas_points_generated = True
                atlas_points_status = "generated"
                atlas_points_reason = "enabled"
            else:
                error_message = (result.stderr or result.stdout or "exporter failed").strip()
                atlas_points_status = "error"
                atlas_points_reason = error_message[:200] or "exporter failed"
        else:
            atlas_points_status = "not_found"
            atlas_points_reason = "exporter not found"
    
    # --- ising_comparison.json ---
    ising_path = outputs_dir / "ising_comparison.json"
    ising_data = {
        "targets": {
            "delta_sigma": cfg.delta_sigma,
            "delta_epsilon": cfg.delta_epsilon,
            "source": "arXiv:2411.15300 (bootstrap precision)",
        },
        "criteria": {
            "tau_delta": cfg.tau_delta,
            "sigma_factor": cfg.sigma_factor,
            "sigma_cap": cfg.sigma_cap,
            "note": "STRONG_PASS: error <= tau. WEAK_PASS: error <= 2*sigma con sigma <= sigma_cap.",
        },
        "sigma_comparison": contracts["c1_sigma"],
        "epsilon_comparison": contracts["c1_epsilon"],
        "interpretation": "Hard-wall baseline: Delta es parametro de barrido, no prediccion. "
                          "C1 valida infraestructura, no fisica. "
                          "Comparacion real requiere geometrias emergentes (Bloque A no trivial).",
    }
    with open(ising_path, "w") as f:
        json.dump(ising_data, f, indent=2)
    
    # --- validation.json ---
    val_path = outputs_dir / "validation.json"
    
    c1_sigma_in_domain = contracts["c1_sigma"].get("in_domain", True)
    c1_epsilon_in_domain = contracts["c1_epsilon"].get("in_domain", True)
    c1_sigma_status = contracts["c1_sigma"]["global_status"]
    c1_epsilon_status = contracts["c1_epsilon"]["global_status"]
    
    c3 = contracts["c3"]
    c3_status = c3.get("status", "SKIP")
    c3_ok = c3.get("spectral_ok", None)
    
    c1_hard_fail = (
        (c1_sigma_in_domain and c1_sigma_status == "FAIL") or
        (c1_epsilon_in_domain and c1_epsilon_status == "FAIL")
    )
    c3_hard_fail = (c3_status != "SKIP" and c3_ok is False)
    
    all_hard_pass = (
        contracts["c2"]["consistency_ok"] and
        not c1_hard_fail and
        not c3_hard_fail
    )
    
    delta_min = float(spectrum["delta_uv"].min())
    delta_max = float(spectrum["delta_uv"].max())
    
    validation = {
        "version": __version__,
        "features": features_meta,
        "metadata": {
            "run": cfg.run,
            "spectrum_source": cfg.spectrum_file,
            "delta_range": [delta_min, delta_max],
            "n_delta": spectrum["n_delta"],
            "n_modes": spectrum["n_modes"],
            "k_features": cfg.k_features,
        },
        "bootstrap": {
            "n_bootstrap": bootstrap["n_bootstrap"],
            "sigma_delta_mean": bootstrap["sigma_delta_mean"],
            "sigma_delta_p90": bootstrap["sigma_delta_p90"],
            "sigma_delta_max": bootstrap["sigma_delta_max"],
        },
        "C1_ising": {
            "description": "Compatibilidad puntual con Ising 3D",
            "sigma_cap": cfg.sigma_cap,
            "note": "OUT_OF_DOMAIN = target fuera del rango; STRONG_PASS = dentro de tau; WEAK_PASS = dentro de 2sigma con sigma<=sigma_cap; FAIL = ninguno",
            "delta_range": [delta_min, delta_max],
            "sigma": contracts["c1_sigma"],
            "epsilon": contracts["c1_epsilon"],
        },
        "C2_consistency": {
            "description": "Consistencia interna del diccionario",
            **contracts["c2"],
        },
        "C3_spectral": c3,
        "C5_exp04": c5_contracts,
        "overall": {
            "C1_sigma_status": c1_sigma_status,
            "C1_sigma_in_domain": c1_sigma_in_domain,
            "C1_epsilon_status": c1_epsilon_status,
            "C1_epsilon_in_domain": c1_epsilon_in_domain,
            "C2_ok": contracts["c2"]["consistency_ok"],
            "C3_status": c3_status,
            "C3_failure_mode": c3.get("failure_mode"),
            "hard_contracts_definition": "C2 siempre; C3 si activo; C1 solo si IN_DOMAIN",
            "all_hard_contracts_pass": all_hard_pass,
        },
    }
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    
    # --- stage_summary.json ---
    summary = {
        "stage": "dictionary",
        "stage_legacy": "04_diccionario",
        "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "run": cfg.run,
        "notes": "El modelo directo Δ→ratios es un proxy suave local usado solo para evaluar invertibilidad (C3). No es física; fallos en C3a indican proxy/modelo insuficiente.",
        "config": {
            "run": cfg.run,
            "spectrum_file": cfg.spectrum_file,
            "export_atlas_points": cfg.export_atlas_points,
            "k_features": cfg.k_features,
            "exp04_compare_run": cfg.exp04_compare_run,
            "models_evaluated": list(cfg.models),
            "cv_folds": cfg.cv_folds,
            "n_bootstrap": cfg.n_bootstrap,
            "random_seed": cfg.random_seed,
            "tau_delta": cfg.tau_delta,
            "sigma_cap": cfg.sigma_cap,
            "enable_c3": cfg.enable_c3,
            "c3_metric": cfg.c3_metric,
            "c3_weights": cfg.c3_weights,
            "c3_threshold": cfg.c3_threshold,
            "c3_adaptive_threshold": cfg.c3_adaptive_threshold,
            "c3_threshold_factor": cfg.c3_threshold_factor,
            "direct_model": cfg.direct_model,
            "direct_degree": cfg.direct_degree,
        },
        "data": {
            "delta_range": [delta_min, delta_max],
            "n_delta": spectrum["n_delta"],
            "n_modes": spectrum["n_modes"],
            "features": features_meta,
        },
        "model": {
            "selected": best_model,
            "selection_criterion": "BIC",
            "n_params": best_result["n_params"],
            "cv_rmse": best_result["cv"]["cv_rmse_mean"],
            "r2": best_result["metrics"]["r2"],
            "aic": best_result["info_criteria"]["aic"],
            "bic": best_result["info_criteria"]["bic"],
        },
        "uncertainty": {
            "sigma_delta_mean": bootstrap["sigma_delta_mean"],
            "sigma_delta_p90": bootstrap["sigma_delta_p90"],
            "sigma_delta_max": bootstrap["sigma_delta_max"],
        },
        "validation_summary": {
            "C1_sigma_status": c1_sigma_status,
            "C1_sigma_in_domain": c1_sigma_in_domain,
            "C1_epsilon_status": c1_epsilon_status,
            "C1_epsilon_in_domain": c1_epsilon_in_domain,
            "C2_consistency_ok": contracts["c2"]["consistency_ok"],
            "C3_status": c3_status,
            "C3_failure_mode": c3.get("failure_mode"),
            "all_hard_contracts_pass": all_hard_pass,
        },
        "hashes": {},
    }

    summary["atlas_points_status"] = atlas_points_status
    summary["atlas_points_reason"] = atlas_points_reason
    summary["atlas_points_generated"] = atlas_points_generated
    if atlas_points_generated:
        summary["atlas_points"] = "outputs/atlas_points.json"
        summary["atlas_points_sha256"] = sha256_file(atlas_points_path)
        summary["atlas_points_feature_key"] = atlas_points_feature_key
    
    summary_path = stage_dir / "stage_summary.json"
    
    summary["hashes"] = {
        "outputs/dictionary.h5": sha256_file(h5_path),
        "outputs/atlas.json": sha256_file(atlas_path),
        "outputs/ising_comparison.json": sha256_file(ising_path),
        "outputs/validation.json": sha256_file(val_path),
    }
    if atlas_points_path.exists():
        summary["hashes"]["outputs/atlas_points.json"] = sha256_file(
            atlas_points_path
        )
    
    write_stage_summary(stage_dir, summary)
    
    # --- manifest.json ---
    manifest_artifacts = {
        "dictionary": h5_path,
        "atlas": atlas_path,
        "ising_comparison": ising_path,
        "validation": val_path,
        "summary": summary_path,
    }
    if atlas_points_path.exists():
        manifest_artifacts["atlas_points"] = atlas_points_path

    manifest_path = write_manifest(
        stage_dir,
        manifest_artifacts,
        extra={
            "version": __version__,
            "stage_legacy": "04_diccionario",
            "input_spectrum": spectrum_rel_path,
            "atlas_points_status": atlas_points_status,
            "atlas_points_reason": atlas_points_reason,
        },
    )
    
    return {
        "dictionary": h5_path,
        "atlas": atlas_path,
        "ising_comparison": ising_path,
        "validation": val_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


# =============================================================================
# Main
# =============================================================================

def run_self_test() -> int:
    """Self-tests deterministas para validar lógica interna de C3.

    T1 (inyectivo): r = Δ -> C3a PASS, C3b PASS
    T2 (no-inyectivo): r = (Δ-5)^2 -> C3a PASS, C3b FAIL/WARN
    """
    print("Running self-test...")
    np.random.seed(0)

    # T1: inyectivo
    Delta = np.linspace(1.0, 10.0, 80)
    X = Delta.reshape(-1, 1)
    cfg = Config(run="__selftest__", test_mode=True, k_features=1, enable_c3=True, direct_model="poly", direct_degree=1)
    res1 = evaluate_c3_full(X, Delta, cfg, inverse_model_type="linear")
    if not (res1.get("c3a_status") == "PASS" and res1.get("c3b_status") == "PASS" and res1.get("status") == "PASS"):
        print(f"FATAL: Self-test T1 fallo (inyectivo). {res1}")
        return 1

    # T2: no-inyectivo (controlado)
    r = (Delta - 5.0) ** 2
    X2 = r.reshape(-1, 1)
    cfg2 = Config(run="__selftest__", test_mode=True, k_features=1, enable_c3=True, direct_model="poly", direct_degree=2)
    res2 = evaluate_c3_full(X2, Delta, cfg2, inverse_model_type="poly2")
    # Esperado: proxy aprende (C3a PASS) pero el ciclo se rompe por ambiguedad (C3b FAIL)
    if not (res2.get("c3a_status") == "PASS" and res2.get("c3b_status") == "FAIL" and res2.get("status") == "FAIL" and res2.get("failure_mode") == "CYCLE_INCONSISTENT"):
        print(f"FATAL: Self-test T2 fallo (no-inyectivo controlado). {res2}")
        return 1

    print("Self-test PASSED.")
    return 0


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    cfg = parse_args()

    if cfg.test_mode:
        return run_self_test()

    if cfg.run is None:
        print("ERROR: --run es requerido", file=sys.stderr)
        return 1
    
    # Cargar espectro
    spec_path = resolve_input_spectrum_path(cfg.run, cfg.spectrum_file)
    if not spec_path.exists():
        print(f"ERROR: No existe espectro en {spec_path}", file=sys.stderr)
        print("       Ejecuta primero: python 03_sturm_liouville.py --run " + cfg.run,
              file=sys.stderr)
        return 1
    
    spectrum = load_spectrum(spec_path)
    run_dir = get_run_dir(cfg.run)
    try:
        spectrum_rel_path = str(spec_path.relative_to(run_dir))
    except ValueError:
        print(f"ERROR: spectrum fuera de runs/{cfg.run}: {spec_path}", file=sys.stderr)
        return 1
    
    print(f"Bloque C v{__version__}")
    print(f"Espectro cargado: {spec_path}")
    print(f"  n_delta={spectrum['n_delta']}, n_modes={spectrum['n_modes']}")
    print(f"  Delta in [{spectrum['delta_uv'].min():.3f}, {spectrum['delta_uv'].max():.3f}]")
    print()
    
    # Validar k_features
    if cfg.k_features >= spectrum["n_modes"]:
        print(f"ERROR: k_features={cfg.k_features} debe ser < n_modes={spectrum['n_modes']}",
              file=sys.stderr)
        return 1
    
    # Preparar datos
    y = spectrum["delta_uv"]
    if spectrum.get("dual_spectrum", False):
        if spectrum["M2_D"].shape != spectrum["M2_N"].shape:
            print("ERROR: M2_D y M2_N deben tener misma forma", file=sys.stderr)
            return 1
    X, features_meta = build_features_from_spectrum(spectrum, cfg.k_features, eps=1e-12)
    
    feature_label = "dual" if features_meta["dual"] else "legacy"
    print(f"Features: {features_meta['dim']} ratios ({feature_label}, k={cfg.k_features})")
    print(f"Dataset: {len(y)} muestras")
    print()
    
    # --- Selección de modelo ---
    print("Evaluando modelos inversos (r -> Delta)...")
    model_selection = select_best_model(X, y, cfg.models, cfg.cv_folds, cfg.random_seed)
    
    for model_type, result in model_selection["all_models"].items():
        marker = "->" if model_type == model_selection["best_model"] else "  "
        print(f"  {marker} {model_type:8s}: BIC={result['info_criteria']['bic']:8.2f}, "
              f"CV RMSE={result['cv']['cv_rmse_mean']:.4f}, R2={result['metrics']['r2']:.4f}")
    
    best_model = model_selection["best_model"]
    best_result = model_selection["all_models"][best_model]
    print(f"\nModelo seleccionado: {best_model} (por BIC)")
    print()
    
    # --- Bootstrap para incertidumbre ---
    print(f"Bootstrap ({cfg.n_bootstrap} replicas)...")
    bootstrap = bootstrap_uncertainty(X, y, best_model, cfg.n_bootstrap, cfg.random_seed)
    print(f"  sigma_Delta medio: {bootstrap['sigma_delta_mean']:.4f}")
    print(f"  sigma_Delta maximo: {bootstrap['sigma_delta_max']:.4f}")
    print()
    
    # --- Predicciones finales ---
    Phi, _ = build_design_matrix(X, best_model)
    beta = fit_model(Phi, y)
    y_pred = predict(Phi, beta)
    
    # --- Contratos ---
    print("Evaluando contratos...")
    
    delta_range = (float(y.min()), float(y.max()))
    
    # C1: Ising
    c1_sigma = evaluate_c1_ising(
        y_pred, bootstrap["sigma_delta"],
        cfg.delta_sigma, cfg.tau_delta, cfg.sigma_factor, "sigma",
        sigma_cap=cfg.sigma_cap, delta_range=delta_range
    )
    c1_epsilon = evaluate_c1_ising(
        y_pred, bootstrap["sigma_delta"],
        cfg.delta_epsilon, cfg.tau_delta, cfg.sigma_factor, "epsilon",
        sigma_cap=cfg.sigma_cap, delta_range=delta_range
    )
    
    # C2: Consistencia
    c2 = evaluate_c2_consistency(y, y_pred, best_result["cv"]["cv_rmse_mean"])
    
    # C3: Espectral (con diagnóstico causal)
    if cfg.enable_c3:
        print(f"  C3: Modelo directo (degree={cfg.direct_degree}), metric={cfg.c3_metric}, weights={cfg.c3_weights}")
    c3 = evaluate_c3_full(X, y, cfg, best_model, sigma_delta_stats=bootstrap)
    
    contracts = {
        "c1_sigma": c1_sigma,
        "c1_epsilon": c1_epsilon,
        "c2": c2,
        "c3": c3,
    }

    current_validation = {
        "C2_consistency": c2,
        "C3_spectral": c3,
    }
    c5_contracts = evaluate_c5(
        current_validation,
        dual=bool(features_meta["dual"]),
        compare_run=cfg.exp04_compare_run,
    )
    
    # Prints
    sigma_status = c1_sigma['global_status']
    epsilon_status = c1_epsilon['global_status']
    
    if sigma_status == "OUT_OF_DOMAIN":
        print(f"  C1 (Ising sigma): OUT_OF_DOMAIN (Delta_sigma={cfg.delta_sigma:.4f} < Delta_min={delta_range[0]:.3f})")
    else:
        print(f"  C1 (Ising sigma): {sigma_status:12s} "
              f"(error: {c1_sigma['best_candidate']['error']:.4f}, "
              f"sigma: {c1_sigma['best_candidate']['sigma_delta']:.4f})")
    
    if epsilon_status == "OUT_OF_DOMAIN":
        print(f"  C1 (Ising epsilon): OUT_OF_DOMAIN (Delta_epsilon={cfg.delta_epsilon:.4f} < Delta_min={delta_range[0]:.3f})")
    else:
        print(f"  C1 (Ising epsilon): {epsilon_status:12s} "
              f"(error: {c1_epsilon['best_candidate']['error']:.4f}, "
              f"sigma: {c1_epsilon['best_candidate']['sigma_delta']:.4f})")
    
    print(f"  C2 (consistencia): {'PASS' if c2['consistency_ok'] else 'FAIL':12s} "
          f"(CV RMSE: {c2['cv_rmse']:.4f})")
    
    c3_status = c3.get("status", "SKIP")
    if c3_status == "SKIP":
        print(f"  C3 (espectral): SKIP (usar --enable-c3)")
    else:
        failure_mode = c3.get("failure_mode", "")
        c3a_val = c3.get("c3a_decoder", {}).get("global", 0)
        c3b_val = c3.get("c3b_cycle", {}).get("global", 0)
        threshold_eff = c3.get("threshold_used", cfg.c3_threshold)
        noise_floor = c3.get("noise_floor", {}).get("median_sigma", 0)
        tol_cycle = c3.get("tol_cycle", 0)
        sensitivity = c3.get("sensitivity", {})
        sigma_delta_used = c3.get("sigma_delta", {}).get("used", 0)
        
        print(f"  C3 (espectral): {c3_status:12s} "
              f"[C3a={c3a_val:.4f}, C3b={c3b_val:.4f}, threshold={threshold_eff:.4f}]")
        if failure_mode:
            print(f"      failure_mode: {failure_mode}")
        print(f"      noise_floor: {noise_floor:.6f}, metric: {cfg.c3_metric}")
        print(f"      tol_cycle: {tol_cycle:.6f}, sigma_delta: {sigma_delta_used:.6f}")
        if sensitivity:
            print(f"      sensitivity: s_p50={sensitivity.get('s_p50', 0):.4f}, "
                  f"s_p90={sensitivity.get('s_p90', 0):.4f}")
        if cfg.c3_weights != "none":
            print(f"      weights: {cfg.c3_weights}")
    
    print()
    
    # --- Atlas ---
    atlas = build_atlas(spectrum["delta_uv"], spectrum["M2"], cfg.k_features)
    
    # --- Escribir outputs ---
    stage_dir, outputs_dir = ensure_stage_dirs(cfg.run, "dictionary")
    paths = write_outputs(
        stage_dir,
        outputs_dir,
        cfg,
        spectrum,
        model_selection,
        bootstrap,
        contracts,
        atlas,
        features_meta,
        spectrum_rel_path,
        c5_contracts,
    )
    
    print("Outputs escritos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print()
    
    # --- Resumen ---
    c3_ok = c3.get("spectral_ok", True) if c3_status != "SKIP" else True
    hard_pass = c2["consistency_ok"] and c3_ok
    
    if hard_pass:
        print("[OK] VERIFICADO: Diccionario consistente")
        if c3_status == "SKIP":
            print("  (C3 no evaluado - usar --enable-c3 para test de ciclo completo)")
        return 0
    else:
        print("[FAIL] FALLO en contratos duros:")
        if not c2["consistency_ok"]:
            print(f"  - C2: CV RMSE = {c2['cv_rmse']:.4f} > 0.05")
        if not c3_ok:
            failure = c3.get("failure_mode", "unknown")
            print(f"  - C3: {failure}")
            if "c3a_decoder" in c3:
                print(f"    C3a (decoder): {c3['c3a_decoder']['global']:.4f}")
            if "c3b_cycle" in c3:
                print(f"    C3b (cycle):   {c3['c3b_cycle']['global']:.4f}")
            print(f"    threshold:     {c3['threshold']['effective']:.4f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
