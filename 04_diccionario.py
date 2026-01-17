#!/usr/bin/env python3
"""
Bloque C: Diccionario holográfico inverso λ_SL → Δ

Aprende la relación inversa: dado un espectro de masas M_n², predice Δ_UV.
Usa features invariantes de escala (ratios r_n = M_n²/M_0²).

Uso:
    python 04_diccionario.py --run mi_experimento
    python 04_diccionario.py --run mi_experimento --k-features 5 --n-bootstrap 500

Salida:
    runs/<run>/dictionary/
        ├── dictionary.h5          (modelo, coeficientes, métricas)
        ├── atlas.json             (clustering de teorías efectivas)
        ├── ising_comparison.json  (contraste con Δ_σ, Δ_ε)
        ├── validation.json        (contratos C1, C2, C3)
        ├── manifest.json
        └── stage_summary.json

Contratos:
    C1: Compatibilidad puntual con Ising (τ_Δ=0.02 o 2σ)
    C2: Consistencia interna (CV RMSE, ciclo Δ→r→Δ)
    C3: Compatibilidad espectral (RMSE en ratios)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

# Silenciar warnings de convergencia en CV con pocos datos
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Configuración
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Configuración del Bloque C."""
    run: str
    spectrum_file: str = "spectrum.h5"
    
    # Features
    k_features: int = 3              # Número de ratios r_n (n=1..K)
    
    # Modelos a evaluar
    models: tuple = ("linear", "poly2")
    
    # Validación
    cv_folds: int = 5
    n_bootstrap: int = 200
    random_seed: int = 42
    
    # Contratos
    tau_delta: float = 0.02          # Tolerancia absoluta para C1
    sigma_factor: float = 2.0        # Factor para criterio 2σ
    sigma_cap: float = 0.1           # Techo para σ usable en C1
    enable_c3: bool = False          # Activar C3 completo (modelo directo)
    
    # C3 tuning (auditable)
    direct_model: str = "poly"       # "linear" o "poly" para modelo directo Î"â†'ratios
    direct_degree: int = 2           # Grado del polinomio si direct_model="poly"
    c3_metric: str = "rmse"          # "rmse" | "rmse_log" | "rmse_rel"
    c3_threshold: float = 0.05       # Umbral configurable para C3
    
    # Targets Ising 3D (bootstrap precision, arXiv:2411.15300)
    delta_sigma: float = 0.518148806
    delta_epsilon: float = 1.41262528


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Bloque C: Diccionario holográfico inverso"
    )
    p.add_argument("--run", type=str, required=True, help="Nombre del run")
    p.add_argument("--spectrum-file", type=str, default="spectrum.h5",
                   dest="spectrum_file", help="Archivo H5 del espectro")
    p.add_argument("--k-features", type=int, default=3, dest="k_features",
                   help="Número de ratios r_n a usar como features (default: 3)")
    p.add_argument("--cv-folds", type=int, default=5, dest="cv_folds")
    p.add_argument("--n-bootstrap", type=int, default=200, dest="n_bootstrap")
    p.add_argument("--seed", type=int, default=42, dest="random_seed")
    p.add_argument("--tau-delta", type=float, default=0.02, dest="tau_delta")
    p.add_argument("--sigma-cap", type=float, default=0.1, dest="sigma_cap",
                   help="Techo máximo de σ para WEAK_PASS en C1 (default: 0.1)")
    p.add_argument("--enable-c3", action="store_true", dest="enable_c3",
                   help="Activar C3 completo con modelo directo")
    p.add_argument("--direct-model", type=str, default="poly", dest="direct_model",
                   choices=["linear", "poly"],
                   help="Modelo directo para C3 (default: poly)")
    p.add_argument("--direct-degree", type=int, default=2, dest="direct_degree",
                   help="Grado del polinomio para el modelo directo (default: 2)")
    p.add_argument("--c3-metric", type=str, default="rmse", dest="c3_metric",
                   choices=["rmse", "rmse_log", "rmse_rel"],
                   help="Metrica para C3 (default: rmse)")
    p.add_argument("--c3-threshold", type=float, default=0.05, dest="c3_threshold",
                   help="Umbral para C3 (default: 0.05)")
    
    args = p.parse_args()
    return Config(
        run=args.run,
        spectrum_file=args.spectrum_file,
        k_features=args.k_features,
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
        data = {
            "delta_uv": h5["delta_uv"][:],
            "m2L2": h5["m2L2"][:],
            "M2": h5["M2"][:],          # shape (n_delta, n_modes)
            "z_grid": h5["z_grid"][:],
            "d": int(h5.attrs["d"]),
            "L": float(h5.attrs["L"]),
            "n_delta": int(h5.attrs["n_delta"]),
            "n_modes": int(h5.attrs["n_modes"]),
        }
    return data


def compute_ratio_features(M2: np.ndarray, k: int) -> np.ndarray:
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
    M0_safe = np.where(np.abs(M0) > 1e-10, M0, 1e-10)
    
    # Ratios r_1, r_2, ..., r_k
    X = M2[:, 1:k+1] / M0_safe
    
    return X


# =============================================================================
# Modelos paramétricos
# =============================================================================

def build_design_matrix(X: np.ndarray, model_type: str) -> np.ndarray:
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
    
    # Log-likelihood (asumiendo errores gaussianos)
    # L = -n/2 * log(2π) - n/2 * log(σ²) - SS_res/(2σ²)
    # Con σ² = SS_res/n (MLE): log(L) = -n/2 * (log(2π) + log(SS_res/n) + 1)
    
    sigma2_mle = ss_res / n
    log_likelihood = -n/2 * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)
    
    # AIC = -2 log(L) + 2k
    # BIC = -2 log(L) + k log(n)
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
        # Índices de test
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        # Split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit
        Phi_train, _ = build_design_matrix(X_train, model_type)
        beta = fit_model(Phi_train, y_train)
        
        # Predict
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
    
    # Ajustar modelo completo para predicciones base
    Phi_full, _ = build_design_matrix(X, model_type)
    beta_full = fit_model(Phi_full, y)
    y_pred_full = predict(Phi_full, beta_full)
    
    # Bootstrap: re-muestreo de filas
    bootstrap_predictions = np.zeros((n_bootstrap, n_samples))
    bootstrap_betas = []
    
    for b in range(n_bootstrap):
        # Muestreo con reemplazo
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_b, y_b = X[idx], y[idx]
        
        # Fit
        Phi_b, _ = build_design_matrix(X_b, model_type)
        beta_b = fit_model(Phi_b, y_b)
        bootstrap_betas.append(beta_b)
        
        # Predict en datos originales
        y_pred_b = predict(Phi_full, beta_b)
        bootstrap_predictions[b] = y_pred_b
    
    # Estadísticas
    sigma_delta = np.std(bootstrap_predictions, axis=0)  # Por punto
    beta_mean = np.mean(bootstrap_betas, axis=0)
    beta_std = np.std(bootstrap_betas, axis=0)
    
    return {
        "sigma_delta": sigma_delta,                    # (n_samples,)
        "sigma_delta_mean": float(np.mean(sigma_delta)),
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
        # Fit completo
        Phi, feature_names = build_design_matrix(X, model_type)
        beta = fit_model(Phi, y)
        y_pred = predict(Phi, beta)
        
        # Métricas
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
    
    # Seleccionar por BIC (menor es mejor)
    best_model = min(results.keys(), key=lambda m: results[m]["info_criteria"]["bic"])
    
    return {
        "all_models": results,
        "best_model": best_model,
        "selection_criterion": "BIC",
    }


# =============================================================================
# Contratos
# =============================================================================

def evaluate_c1_ising(delta_pred: np.ndarray, sigma_delta: np.ndarray,
                      delta_target: float, tau: float, sigma_factor: float,
                      target_name: str, sigma_cap: float = 0.1,
                      delta_range: tuple[float, float] = None) -> dict:
    """Contrato C1: compatibilidad puntual con Ising.
    
    Criterios:
    - OUT_OF_DOMAIN: target fuera del rango [delta_min, delta_max] del dataset
    - STRONG_PASS: |Δ̂ - Δ*| ≤ τ (tolerancia absoluta)
    - WEAK_PASS: |Δ̂ - Δ*| ≤ 2σ AND σ ≤ sigma_cap (incertidumbre acotada)
    - FAIL: ninguno de los anteriores
    
    El sigma_cap evita "pasar por incertidumbre gigante".
    """
    errors = np.abs(delta_pred - delta_target)
    
    # Verificar si el target está en dominio
    in_domain = True
    if delta_range is not None:
        delta_min, delta_max = delta_range
        in_domain = delta_min <= delta_target <= delta_max
    
    # Si está fuera de dominio, no evaluamos pass/fail
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
    
    # Criterio fuerte: tolerancia absoluta
    pass_tau = errors <= tau
    
    # Criterio débil: dentro de 2σ, pero solo si σ es razonable
    sigma_usable = sigma_delta <= sigma_cap
    pass_sigma_raw = errors <= sigma_factor * sigma_delta
    pass_sigma = pass_sigma_raw & sigma_usable
    
    # Mejor candidato: primero por pass_tau, luego por pass_sigma, luego por error mínimo
    # Construir score: 0 si pass_tau, 1 si pass_sigma, 2 si ninguno, luego desempatar por error
    scores = np.where(pass_tau, 0, np.where(pass_sigma, 1, 2))
    # Combinar score con error normalizado para desempate
    combined_score = scores + errors / (errors.max() + 1e-10)
    best_idx = np.argmin(combined_score)
    
    # Status del mejor candidato
    if pass_tau[best_idx]:
        status = "STRONG_PASS"
    elif pass_sigma[best_idx]:
        status = "WEAK_PASS"
    elif pass_sigma_raw[best_idx] and not sigma_usable[best_idx]:
        status = "FAIL_SIGMA_TOO_LARGE"
    else:
        status = "FAIL"
    
    # Status global
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
    
    # Criterio de fallo: CV RMSE > 0.05
    consistency_ok = cv_rmse < 0.05
    
    return {
        "fit_rmse": metrics["rmse"],
        "fit_mae": metrics["mae"],
        "fit_r2": metrics["r2"],
        "cv_rmse": cv_rmse,
        "threshold_cv_rmse": 0.05,
        "consistency_ok": consistency_ok,
    }


def fit_direct_model(y: np.ndarray, X: np.ndarray, model_type: str, degree: int = 2) -> tuple[np.ndarray, list]:
    """Ajusta modelo directo Delta -> ratios (para C3).
    
    Args:
        y: array (n_samples,) con Delta
        X: array (n_samples, k) con ratios objetivo
        model_type: "linear" o "poly"
        degree: grado del polinomio si model_type="poly"
    
    Returns:
        betas: lista de coeficientes, uno por ratio
        feature_names: nombres de features del modelo
    """
    n_samples, k = X.shape
    betas = []
    
    # Para el modelo directo, las features son funciones de Delta
    if model_type == "linear":
        Phi = np.column_stack([np.ones(n_samples), y])
        names = ["1", "Delta"]
    elif model_type == "poly":
        # [1, Delta, Delta^2, ..., Delta^degree]
        cols = [np.ones(n_samples)] + [y**p for p in range(1, degree + 1)]
        Phi = np.column_stack(cols)
        names = ["1"] + [f"Delta^{p}" for p in range(1, degree + 1)]
    else:
        raise ValueError(f"model_type desconocido: {model_type}")
    
    # Ajustar un modelo por cada ratio
    for j in range(k):
        beta_j = fit_model(Phi, X[:, j])
        betas.append(beta_j)
    
    return betas, names


def predict_ratios(y: np.ndarray, betas: list, model_type: str, degree: int = 2) -> np.ndarray:
    """Predice ratios desde Delta usando modelo directo."""
    n_samples = len(y)
    k = len(betas)
    
    if model_type == "linear":
        Phi = np.column_stack([np.ones(n_samples), y])
    elif model_type == "poly":
        cols = [np.ones(n_samples)] + [y**p for p in range(1, degree + 1)]
        Phi = np.column_stack(cols)
    else:
        raise ValueError(f"model_type desconocido: {model_type}")
    
    X_pred = np.zeros((n_samples, k))
    for j, beta_j in enumerate(betas):
        X_pred[:, j] = Phi @ beta_j
    
    return X_pred


def evaluate_c3_spectral(X: np.ndarray, X_pred: np.ndarray,
                         metric: str = "rmse", threshold: float = 0.05,
                         is_placeholder: bool = False) -> dict:
    """Contrato C3: compatibilidad espectral (error en ratios).
    
    Args:
        X: ratios originales
        X_pred: ratios reconstruidos
        metric: "rmse" | "rmse_log" | "rmse_rel"
        threshold: umbral para PASS/FAIL
        is_placeholder: si True, retorna SKIP
    
    Si is_placeholder=True, marca como SKIP en lugar de evaluar.
    """
    if is_placeholder:
        return {
            "status": "SKIP",
            "rmse_per_ratio": [],
            "rmse_global": None,
            "spectral_ok": None,
            "metric": metric,
            "threshold": threshold,
            "note": "C3 requiere modelo directo. Use --enable-c3 para activar.",
        }
    
    eps = 1e-12
    
    # Calcular errores segun metrica
    if metric == "rmse":
        errors = X - X_pred
    elif metric == "rmse_log":
        # RMSE sobre log(r) - estabiliza ratios grandes
        X_safe = np.clip(X, eps, None)
        X_pred_safe = np.clip(X_pred, eps, None)
        errors = np.log(X_safe) - np.log(X_pred_safe)
    elif metric == "rmse_rel":
        # RMSE relativo
        denom = np.clip(np.abs(X), eps, None)
        errors = (X - X_pred) / denom
    else:
        raise ValueError(f"metric desconocida: {metric}")
    
    rmse_per_ratio = np.sqrt(np.mean(errors ** 2, axis=0))
    rmse_global = np.sqrt(np.mean(errors ** 2))
    spectral_ok = bool(rmse_global < threshold)
    
    return {
        "status": "PASS" if spectral_ok else "FAIL",
        "metric": metric,
        "rmse_per_ratio": [float(r) for r in rmse_per_ratio],
        "rmse_global": float(rmse_global),
        "threshold": threshold,
        "spectral_ok": spectral_ok,
    }


def evaluate_c3_cycle(X: np.ndarray, y: np.ndarray, 
                      inverse_model_type: str,
                      direct_model_type: str = "poly",
                      direct_degree: int = 2,
                      c3_metric: str = "rmse",
                      c3_threshold: float = 0.05) -> dict:
    """Contrato C3 completo: test de ciclo r -> Delta_hat -> r_hat.
    
    1. Ajusta modelo inverso: r -> Delta
    2. Ajusta modelo directo: Delta -> r
    3. Evalua ciclo: r -> Delta_hat -> r_hat
    4. Mide error(r, r_hat) segun metrica
    
    Args:
        X: ratios originales
        y: Delta (target del modelo inverso)
        inverse_model_type: tipo de modelo inverso ("linear", "poly2")
        direct_model_type: tipo de modelo directo ("linear", "poly")
        direct_degree: grado del polinomio para modelo directo
        c3_metric: metrica de error ("rmse", "rmse_log", "rmse_rel")
        c3_threshold: umbral para PASS/FAIL
    """
    # Modelo inverso (ya lo tenemos, pero lo re-ajustamos para claridad)
    Phi_inv, _ = build_design_matrix(X, inverse_model_type)
    beta_inv = fit_model(Phi_inv, y)
    delta_pred = predict(Phi_inv, beta_inv)
    
    # Modelo directo
    betas_dir, _ = fit_direct_model(y, X, direct_model_type, direct_degree)
    
    # Ciclo: r -> Delta_hat -> r_hat
    X_reconstructed = predict_ratios(delta_pred, betas_dir, direct_model_type, direct_degree)
    
    # Metricas del ciclo
    return evaluate_c3_spectral(X, X_reconstructed, 
                                metric=c3_metric, threshold=c3_threshold,
                                is_placeholder=False)


# =============================================================================
# Atlas de teorías efectivas
# =============================================================================

def build_atlas(delta: np.ndarray, M2: np.ndarray, k: int) -> dict:
    """Construye atlas simple de teorías efectivas.
    
    En hard-wall es trivial (cada Δ define una teoría), pero dejamos
    el formato preparado para geometrías más complejas.
    """
    X = compute_ratio_features(M2, k)
    
    # Clustering simple por Δ (ya están ordenados)
    n_theories = len(delta)
    
    # Estadísticas por teoría
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
        "clustering_method": "parametric_sweep",  # Trivial para hard-wall
    }


# =============================================================================
# IO
# =============================================================================

def compute_file_hash(path: Path) -> str:
    """Calcula SHA256 de un archivo."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_outputs(out_dir: Path, cfg: Config, spectrum: dict,
                  model_selection: dict, bootstrap: dict,
                  contracts: dict, atlas: dict) -> dict:
    """Escribe todos los outputs del Bloque C."""
    import h5py
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_model = model_selection["best_model"]
    best_result = model_selection["all_models"][best_model]
    
    # --- dictionary.h5 ---
    h5_path = out_dir / "dictionary.h5"
    
    with h5py.File(h5_path, "w") as h5:
        # Features
        feat_grp = h5.create_group("features")
        feat_grp.attrs["k"] = cfg.k_features
        feat_grp.attrs["definition"] = "r_n = M_n^2 / M_0^2, n=1..k"
        feat_grp.create_dataset("feature_names", 
                                data=np.array(best_result["feature_names"], dtype="S"))
        
        # Model
        model_grp = h5.create_group("model")
        model_grp.attrs["type"] = best_model
        model_grp.attrs["n_params"] = best_result["n_params"]
        model_grp.create_dataset("beta", data=best_result["beta"])
        model_grp.create_dataset("beta_bootstrap_mean", data=bootstrap["beta_mean"])
        model_grp.create_dataset("beta_bootstrap_std", data=bootstrap["beta_std"])
        
        # Selection
        sel_grp = h5.create_group("selection")
        sel_grp.attrs["criterion"] = "BIC"
        sel_grp.attrs["best_model"] = best_model
        for model_type, result in model_selection["all_models"].items():
            m_grp = sel_grp.create_group(model_type)
            m_grp.attrs["aic"] = result["info_criteria"]["aic"]
            m_grp.attrs["bic"] = result["info_criteria"]["bic"]
            m_grp.attrs["cv_rmse"] = result["cv"]["cv_rmse_mean"]
            m_grp.attrs["r2"] = result["metrics"]["r2"]
        
        # Uncertainty
        unc_grp = h5.create_group("uncertainty")
        unc_grp.attrs["method"] = "bootstrap"
        unc_grp.attrs["n_bootstrap"] = bootstrap["n_bootstrap"]
        unc_grp.attrs["seed"] = cfg.random_seed
        unc_grp.create_dataset("sigma_delta", data=bootstrap["sigma_delta"])
        
        # Predictions (para reproducibilidad)
        pred_grp = h5.create_group("predictions")
        pred_grp.create_dataset("delta_true", data=spectrum["delta_uv"])
        pred_grp.create_dataset("delta_pred", 
                                data=best_result["cv"]["cv_predictions"])
        
        # Metadata
        h5.attrs["created"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["spectrum_source"] = cfg.spectrum_file
    
    # --- atlas.json ---
    atlas_path = out_dir / "atlas.json"
    with open(atlas_path, "w") as f:
        json.dump(atlas, f, indent=2)
    
    # --- ising_comparison.json ---
    ising_path = out_dir / "ising_comparison.json"
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
            "note": "STRONG_PASS requiere error ≤ τ. WEAK_PASS requiere error ≤ 2σ con σ ≤ σ_cap.",
        },
        "sigma_comparison": contracts["c1_sigma"],
        "epsilon_comparison": contracts["c1_epsilon"],
        "interpretation": "Hard-wall baseline: Δ es parámetro de barrido, no predicción. "
                          "C1 aquí valida infraestructura, no física. "
                          "Comparación real requiere geometrías emergentes (Bloque A no trivial).",
    }
    with open(ising_path, "w") as f:
        json.dump(ising_data, f, indent=2)
    
    # --- validation.json ---
    val_path = out_dir / "validation.json"
    
    # Determinar status de C3
    c3_ok = contracts["c3"]["spectral_ok"] if contracts["c3"]["spectral_ok"] is not None else None
    c3_status = contracts["c3"]["status"]
    
    # Lógica de hard contracts:
    # - C2 siempre es hard
    # - C3 es hard solo si está activo (no SKIP)
    # - C1 es hard solo si IN_DOMAIN (OUT_OF_DOMAIN no cuenta)
    c1_sigma_in_domain = contracts["c1_sigma"].get("in_domain", True)
    c1_epsilon_in_domain = contracts["c1_epsilon"].get("in_domain", True)
    c1_sigma_status = contracts["c1_sigma"]["global_status"]
    c1_epsilon_status = contracts["c1_epsilon"]["global_status"]
    
    # C1 hard fail solo si está in_domain y FAIL
    c1_hard_fail = (
        (c1_sigma_in_domain and c1_sigma_status == "FAIL") or
        (c1_epsilon_in_domain and c1_epsilon_status == "FAIL")
    )
    
    # C3 hard fail solo si está activo y FAIL
    c3_hard_fail = (c3_status != "SKIP" and c3_ok is False)
    
    all_hard_pass = (
        contracts["c2"]["consistency_ok"] and
        not c1_hard_fail and
        not c3_hard_fail
    )
    
    # Rango de Δ del dataset
    delta_min = float(spectrum["delta_uv"].min())
    delta_max = float(spectrum["delta_uv"].max())
    
    validation = {
        "metadata": {
            "run": cfg.run,
            "spectrum_source": cfg.spectrum_file,
            "delta_range": [delta_min, delta_max],
            "n_delta": spectrum["n_delta"],
            "n_modes": spectrum["n_modes"],
            "k_features": cfg.k_features,
        },
        "C1_ising": {
            "description": "Compatibilidad puntual con Ising 3D",
            "sigma_cap": cfg.sigma_cap,
            "note": "OUT_OF_DOMAIN = target fuera del rango; STRONG_PASS = dentro de τ; WEAK_PASS = dentro de 2σ con σ≤σ_cap; FAIL = ninguno",
            "delta_range": [delta_min, delta_max],
            "sigma": contracts["c1_sigma"],
            "epsilon": contracts["c1_epsilon"],
        },
        "C2_consistency": {
            "description": "Consistencia interna del diccionario",
            **contracts["c2"],
        },
        "C3_spectral": {
            "description": "Compatibilidad espectral (test de ciclo r→Δ→r)",
            **contracts["c3"],
        },
        "overall": {
            "C1_sigma_status": c1_sigma_status,
            "C1_sigma_in_domain": c1_sigma_in_domain,
            "C1_epsilon_status": c1_epsilon_status,
            "C1_epsilon_in_domain": c1_epsilon_in_domain,
            "C2_ok": contracts["c2"]["consistency_ok"],
            "C3_status": c3_status,
            "hard_contracts_definition": "C2 siempre; C3 si activo; C1 solo si IN_DOMAIN",
            "all_hard_contracts_pass": all_hard_pass,
        },
    }
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    
    # --- stage_summary.json ---
    
    # Re-calcular lógica de contratos para summary
    c1_sigma_in_domain = contracts["c1_sigma"].get("in_domain", True)
    c1_epsilon_in_domain = contracts["c1_epsilon"].get("in_domain", True)
    c1_sigma_status_sum = contracts["c1_sigma"]["global_status"]
    c1_epsilon_status_sum = contracts["c1_epsilon"]["global_status"]
    c3_ok_sum = contracts["c3"]["spectral_ok"] if contracts["c3"]["spectral_ok"] is not None else None
    c3_status_sum = contracts["c3"]["status"]
    
    c1_hard_fail_sum = (
        (c1_sigma_in_domain and c1_sigma_status_sum == "FAIL") or
        (c1_epsilon_in_domain and c1_epsilon_status_sum == "FAIL")
    )
    c3_hard_fail_sum = (c3_status_sum != "SKIP" and c3_ok_sum is False)
    
    all_hard_pass_sum = (
        contracts["c2"]["consistency_ok"] and
        not c1_hard_fail_sum and
        not c3_hard_fail_sum
    )
    
    summary = {
        "stage": "04_diccionario",
        "version": "1.3.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": {
            "run": cfg.run,
            "spectrum_file": cfg.spectrum_file,
            "k_features": cfg.k_features,
            "models_evaluated": list(cfg.models),
            "cv_folds": cfg.cv_folds,
            "n_bootstrap": cfg.n_bootstrap,
            "random_seed": cfg.random_seed,
            "tau_delta": cfg.tau_delta,
            "sigma_cap": cfg.sigma_cap,
            "enable_c3": cfg.enable_c3,
            "direct_model": cfg.direct_model,
            "direct_degree": cfg.direct_degree,
            "c3_metric": cfg.c3_metric,
            "c3_threshold": cfg.c3_threshold,
        },
        "data": {
            "delta_range": [float(spectrum["delta_uv"].min()), float(spectrum["delta_uv"].max())],
            "n_delta": spectrum["n_delta"],
            "n_modes": spectrum["n_modes"],
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
            "sigma_delta_max": bootstrap["sigma_delta_max"],
        },
        "validation_summary": {
            "C1_sigma_status": c1_sigma_status_sum,
            "C1_sigma_in_domain": c1_sigma_in_domain,
            "C1_epsilon_status": c1_epsilon_status_sum,
            "C1_epsilon_in_domain": c1_epsilon_in_domain,
            "C2_consistency_ok": contracts["c2"]["consistency_ok"],
            "C3_status": c3_status_sum,
            "all_hard_contracts_pass": all_hard_pass_sum,
        },
        "hashes": {},  # Se llena después
    }
    
    summary_path = out_dir / "stage_summary.json"
    
    # Calcular hashes
    summary["hashes"] = {
        "dictionary.h5": compute_file_hash(h5_path),
        "atlas.json": compute_file_hash(atlas_path),
        "ising_comparison.json": compute_file_hash(ising_path),
        "validation.json": compute_file_hash(val_path),
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # --- manifest.json ---
    manifest = {
        "stage": "04_diccionario",
        "run": cfg.run,
        "created": datetime.now(timezone.utc).isoformat(),
        "files": {
            "dictionary": "dictionary.h5",
            "atlas": "atlas.json",
            "ising_comparison": "ising_comparison.json",
            "validation": "validation.json",
            "summary": "stage_summary.json",
        },
        "input_spectrum": f"../spectrum/{cfg.spectrum_file}",
    }
    
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
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

def main() -> int:
    cfg = parse_args()
    
    # Cargar espectro
    spec_path = Path("runs") / cfg.run / "spectrum" / cfg.spectrum_file
    if not spec_path.exists():
        print(f"ERROR: No existe espectro en {spec_path}", file=sys.stderr)
        print("       Ejecuta primero: python 03_sturm_liouville.py --run " + cfg.run,
              file=sys.stderr)
        return 1
    
    spectrum = load_spectrum(spec_path)
    
    print(f"Espectro cargado: {spec_path}")
    print(f"  n_delta={spectrum['n_delta']}, n_modes={spectrum['n_modes']}")
    print(f"  Δ ∈ [{spectrum['delta_uv'].min():.3f}, {spectrum['delta_uv'].max():.3f}]")
    print()
    
    # Validar k_features
    if cfg.k_features >= spectrum["n_modes"]:
        print(f"ERROR: k_features={cfg.k_features} debe ser < n_modes={spectrum['n_modes']}",
              file=sys.stderr)
        return 1
    
    # Preparar datos
    y = spectrum["delta_uv"]
    X = compute_ratio_features(spectrum["M2"], cfg.k_features)
    
    print(f"Features: {cfg.k_features} ratios (r_1, ..., r_{cfg.k_features})")
    print(f"Dataset: {len(y)} muestras")
    print()
    
    # --- Selección de modelo ---
    print("Evaluando modelos...")
    model_selection = select_best_model(X, y, cfg.models, cfg.cv_folds, cfg.random_seed)
    
    for model_type, result in model_selection["all_models"].items():
        marker = "→" if model_type == model_selection["best_model"] else " "
        print(f"  {marker} {model_type:8s}: BIC={result['info_criteria']['bic']:8.2f}, "
              f"CV RMSE={result['cv']['cv_rmse_mean']:.4f}, R²={result['metrics']['r2']:.4f}")
    
    best_model = model_selection["best_model"]
    best_result = model_selection["all_models"][best_model]
    print(f"\nModelo seleccionado: {best_model} (por BIC)")
    print()
    
    # --- Bootstrap para incertidumbre ---
    print(f"Bootstrap ({cfg.n_bootstrap} réplicas)...")
    bootstrap = bootstrap_uncertainty(X, y, best_model, cfg.n_bootstrap, cfg.random_seed)
    print(f"  σ_Δ medio: {bootstrap['sigma_delta_mean']:.4f}")
    print(f"  σ_Δ máximo: {bootstrap['sigma_delta_max']:.4f}")
    print()
    
    # --- Predicciones finales ---
    Phi, _ = build_design_matrix(X, best_model)
    beta = fit_model(Phi, y)
    y_pred = predict(Phi, beta)
    
    # --- Contratos ---
    print("Evaluando contratos...")
    
    # Rango de Δ del dataset (para detectar OUT_OF_DOMAIN)
    delta_range = (float(y.min()), float(y.max()))
    
    # C1: Ising (con sigma_cap y delta_range)
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
    
    # C3: Espectral (completo si --enable-c3, placeholder si no)
    if cfg.enable_c3:
        print(f"  C3: Evaluando modelo directo (model={cfg.direct_model}, degree={cfg.direct_degree}, metric={cfg.c3_metric})...")
        c3 = evaluate_c3_cycle(
            X, y, best_model,
            direct_model_type=cfg.direct_model,
            direct_degree=cfg.direct_degree,
            c3_metric=cfg.c3_metric,
            c3_threshold=cfg.c3_threshold
        )
    else:
        c3 = evaluate_c3_spectral(X, X, metric=cfg.c3_metric, threshold=cfg.c3_threshold, is_placeholder=True)

    contracts = {
        "c1_sigma": c1_sigma,
        "c1_epsilon": c1_epsilon,
        "c2": c2,
        "c3": c3,
    }
    
    # Prints mejorados con status
    sigma_status = c1_sigma['global_status']
    epsilon_status = c1_epsilon['global_status']
    
    if sigma_status == "OUT_OF_DOMAIN":
        print(f"  C1 (Ising σ): OUT_OF_DOMAIN (Δ_σ={cfg.delta_sigma:.4f} < Δ_min={delta_range[0]:.3f})")
    else:
        print(f"  C1 (Ising σ): {sigma_status:12s} "
              f"(error: {c1_sigma['best_candidate']['error']:.4f}, "
              f"σ: {c1_sigma['best_candidate']['sigma_delta']:.4f})")
    
    if epsilon_status == "OUT_OF_DOMAIN":
        print(f"  C1 (Ising ε): OUT_OF_DOMAIN (Δ_ε={cfg.delta_epsilon:.4f} < Δ_min={delta_range[0]:.3f})")
    else:
        print(f"  C1 (Ising ε): {epsilon_status:12s} "
              f"(error: {c1_epsilon['best_candidate']['error']:.4f}, "
              f"σ: {c1_epsilon['best_candidate']['sigma_delta']:.4f})")
    
    print(f"  C2 (consistencia): {'PASS' if c2['consistency_ok'] else 'FAIL':12s} "
          f"(CV RMSE: {c2['cv_rmse']:.4f})")
    print(f"  C3 (espectral): {c3['status']:12s}"
          + (f" (RMSE: {c3['rmse_global']:.4f})" if c3['rmse_global'] is not None else ""))
    print()
    
    # --- Atlas ---
    atlas = build_atlas(spectrum["delta_uv"], spectrum["M2"], cfg.k_features)
    
    # --- Escribir outputs ---
    out_dir = Path("runs") / cfg.run / "dictionary"
    paths = write_outputs(out_dir, cfg, spectrum, model_selection, bootstrap, contracts, atlas)
    
    print("Outputs escritos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print()
    
    # --- Resumen ---
    c3_status = contracts["c3"]["status"]
    c3_ok = contracts["c3"]["spectral_ok"]
    
    # Resultado final basado en contratos duros (C2, C3 si está activo)
    hard_pass = c2["consistency_ok"] and (c3_ok if c3_ok is not None else True)
    
    if hard_pass:
        print("✓ VERIFICADO: Diccionario consistente")
        if c3_status == "SKIP":
            print("  (C3 no evaluado - usar --enable-c3 para test de ciclo completo)")
        return 0
    else:
        print("✗ FALLO en contratos duros:")
        if not c2["consistency_ok"]:
            print(f"  - C2: CV RMSE = {c2['cv_rmse']:.4f} > 0.05")
        if c3_ok is not None and not c3_ok:
            print(f"  - C3: RMSE ciclo = {contracts['c3']['rmse_global']:.4f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
