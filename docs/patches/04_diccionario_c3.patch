[1mdiff --git a/04_diccionario.py b/04_diccionario.py[m
[1mindex 34b02fd..7be81d7 100644[m
[1m--- a/04_diccionario.py[m
[1m+++ b/04_diccionario.py[m
[36m@@ -2,12 +2,15 @@[m
 """[m
 Bloque C: Diccionario holográfico inverso λ_SL → Δ[m
 [m
[32m+[m[32mv1.4.0 - Diagnóstico causal C3 con separación C3a/C3b, pesos, noise floor adaptativo[m
[32m+[m
 Aprende la relación inversa: dado un espectro de masas M_n², predice Δ_UV.[m
 Usa features invariantes de escala (ratios r_n = M_n²/M_0²).[m
 [m
 Uso:[m
     python 04_diccionario.py --run mi_experimento[m
[31m-    python 04_diccionario.py --run mi_experimento --k-features 5 --n-bootstrap 500[m
[32m+[m[32m    python 04_diccionario.py --run mi_experimento --enable-c3[m
[32m+[m[32m    python 04_diccionario.py --run mi_experimento --enable-c3 --c3-weights inv_n4 --c3-adaptive-threshold on[m
 [m
 Salida:[m
     runs/<run>/dictionary/[m
[36m@@ -21,7 +24,9 @@[m [mSalida:[m
 Contratos:[m
     C1: Compatibilidad puntual con Ising (τ_Δ=0.02 o 2σ)[m
     C2: Consistencia interna (CV RMSE, ciclo Δ→r→Δ)[m
[31m-    C3: Compatibilidad espectral (RMSE en ratios)[m
[32m+[m[32m    C3: Compatibilidad espectral con diagnóstico causal[m
[32m+[m[32m        C3a (decoder): evalúa modelo directo Δ→r[m
[32m+[m[32m        C3b (cycle): evalúa ciclo completo r→Δ̂→r̂[m
 """[m
 [m
 from __future__ import annotations[m
[36m@@ -31,16 +36,18 @@[m [mimport hashlib[m
 import json[m
 import sys[m
 import warnings[m
[31m-from dataclasses import dataclass, asdict[m
[32m+[m[32mfrom dataclasses import dataclass, asdict, field[m
 from datetime import datetime, timezone[m
 from pathlib import Path[m
[31m-from typing import Literal[m
[32m+[m[32mfrom typing import Literal, Optional[m
 [m
 import numpy as np[m
 [m
 # Silenciar warnings de convergencia en CV con pocos datos[m
 warnings.filterwarnings("ignore", category=UserWarning)[m
 [m
[32m+[m[32m__version__ = "1.4.1"[m
[32m+[m
 [m
 # =============================================================================[m
 # Configuración[m
[36m@@ -50,30 +57,43 @@[m [mwarnings.filterwarnings("ignore", category=UserWarning)[m
 class Config:[m
     """Configuración del Bloque C."""[m
     run: str[m
[32m+[m[32m    test_mode: bool = False  # Ejecuta self-tests deterministas y sale[m
     spectrum_file: str = "spectrum.h5"[m
     [m
     # Features[m
     k_features: int = 3              # Número de ratios r_n (n=1..K)[m
     [m
[31m-    # Modelos a evaluar[m
[32m+[m[32m    # Modelos a evaluar (inverso: r→Δ)[m
     models: tuple = ("linear", "poly2")[m
     [m
[32m+[m[32m    # Modelo directo (Δ→r) para C3[m
[32m+[m[32m    direct_model: str = "poly"       # "linear" o "poly"[m
[32m+[m[32m    direct_degree: int = 4           # Grado del polinomio directo[m
[32m+[m[41m    [m
     # Validación[m
     cv_folds: int = 5[m
     n_bootstrap: int = 200[m
     random_seed: int = 42[m
     [m
[31m-    # Contratos[m
[32m+[m[32m    # Contratos C1[m
     tau_delta: float = 0.02          # Tolerancia absoluta para C1[m
     sigma_factor: float = 2.0        # Factor para criterio 2σ[m
     sigma_cap: float = 0.1           # Techo para σ usable en C1[m
[31m-    enable_c3: bool = False          # Activar C3 completo (modelo directo)[m
     [m
[31m-    # C3 tuning (auditable)[m
[31m-    direct_model: str = "poly"       # "linear" o "poly" para modelo directo Î"â†'ratios[m
[31m-    direct_degree: int = 2           # Grado del polinomio si direct_model="poly"[m
[31m-    c3_metric: str = "rmse"          # "rmse" | "rmse_log" | "rmse_rel"[m
[31m-    c3_threshold: float = 0.05       # Umbral configurable para C3[m
[32m+[m[32m    # Contrato C3[m
[32m+[m[32m    enable_c3: bool = False          # Activar C3 completo[m
[32m+[m[32m    c3_metric: str = "rmse"          # rmse, rmse_log, rmse_rel[m
[32m+[m[32m    c3_threshold: float = 0.05       # Umbral base para C3[m
[32m+[m[32m    c3_weights: str = "none"         # none, inv_n, inv_n2, inv_n4, inv_r2[m
[32m+[m[41m    [m
[32m+[m[32m    # C3 Noise floor adaptativo[m
[32m+[m[32m    c3_noise_floor_eps: float = 0.001[m
[32m+[m[32m    c3_noise_floor_metric: Optional[str] = None  # None = igual a c3_metric[m
[32m+[m[32m    c3_adaptive_threshold: bool = False[m
[32m+[m[32m    c3_threshold_factor: float = 5.0[m
[32m+[m[41m    [m
[32m+[m[32m    # C3 Oracle test[m
[32m+[m[32m    c3_oracle: bool = False[m
     [m
     # Targets Ising 3D (bootstrap precision, arXiv:2411.15300)[m
     delta_sigma: float = 0.518148806[m
[36m@@ -82,9 +102,11 @@[m [mclass Config:[m
 [m
 def parse_args() -> Config:[m
     p = argparse.ArgumentParser([m
[31m-        description="Bloque C: Diccionario holográfico inverso"[m
[32m+[m[32m        description="Bloque C: Diccionario holográfico inverso (v1.4.1)"[m
     )[m
[31m-    p.add_argument("--run", type=str, required=True, help="Nombre del run")[m
[32m+[m[32m    p.add_argument("--run", type=str, default=None, help="Nombre del run")[m
[32m+[m[32m    p.add_argument("--test", action="store_true", dest="test_mode",[m
[32m+[m[32m                   help="Ejecuta self-tests deterministas (T1/T2) y sale")[m
     p.add_argument("--spectrum-file", type=str, default="spectrum.h5",[m
                    dest="spectrum_file", help="Archivo H5 del espectro")[m
     p.add_argument("--k-features", type=int, default=3, dest="k_features",[m
[36m@@ -95,22 +117,50 @@[m [mdef parse_args() -> Config:[m
     p.add_argument("--tau-delta", type=float, default=0.02, dest="tau_delta")[m
     p.add_argument("--sigma-cap", type=float, default=0.1, dest="sigma_cap",[m
                    help="Techo máximo de σ para WEAK_PASS en C1 (default: 0.1)")[m
[32m+[m[41m    [m
[32m+[m[32m    # C3 básico[m
     p.add_argument("--enable-c3", action="store_true", dest="enable_c3",[m
[31m-                   help="Activar C3 completo con modelo directo")[m
[32m+[m[32m                   help="Activar C3 completo con modelo directo Δ→ratios")[m
     p.add_argument("--direct-model", type=str, default="poly", dest="direct_model",[m
[31m-                   choices=["linear", "poly"],[m
[31m-                   help="Modelo directo para C3 (default: poly)")[m
[31m-    p.add_argument("--direct-degree", type=int, default=2, dest="direct_degree",[m
[31m-                   help="Grado del polinomio para el modelo directo (default: 2)")[m
[32m+[m[32m                   choices=["linear", "poly"], help="Tipo de modelo directo (default: poly)")[m
[32m+[m[32m    p.add_argument("--direct-degree", type=int, default=4, dest="direct_degree",[m
[32m+[m[32m                   help="Grado del polinomio para modelo directo (default: 4)")[m
     p.add_argument("--c3-metric", type=str, default="rmse", dest="c3_metric",[m
                    choices=["rmse", "rmse_log", "rmse_rel"],[m
[31m-                   help="Metrica para C3 (default: rmse)")[m
[32m+[m[32m                   help="Métrica para C3 (default: rmse)")[m
     p.add_argument("--c3-threshold", type=float, default=0.05, dest="c3_threshold",[m
[31m-                   help="Umbral para C3 (default: 0.05)")[m
[32m+[m[32m                   help="Umbral base para C3 (default: 0.05)")[m
[32m+[m[41m    [m
[32m+[m[32m    # C3 pesos[m
[32m+[m[32m    p.add_argument("--c3-weights", type=str, default="none", dest="c3_weights",[m
[32m+[m[32m                   choices=["none", "inv_n", "inv_n2", "inv_n4", "inv_r2"],[m
[32m+[m[32m                   help="Esquema de pesos para C3 (default: none)")[m
[32m+[m[41m    [m
[32m+[m[32m    # C3 noise floor adaptativo[m
[32m+[m[32m    p.add_argument("--c3-noise-floor-eps", type=float, default=0.001,[m
[32m+[m[32m                   dest="c3_noise_floor_eps",[m
[32m+[m[32m                   help="Epsilon para cálculo de noise floor (default: 0.001)")[m
[32m+[m[32m    p.add_argument("--c3-noise-floor-metric", type=str, default=None,[m
[32m+[m[32m                   dest="c3_noise_floor_metric",[m
[32m+[m[32m                   choices=["rmse", "rmse_log", "rmse_rel", None],[m
[32m+[m[32m                   help="Métrica para noise floor (default: igual a --c3-metric)")[m
[32m+[m[32m    p.add_argument("--c3-adaptive-threshold", action="store_true",[m
[32m+[m[32m                   dest="c3_adaptive_threshold",[m
[32m+[m[32m                   help="Usar umbral adaptativo basado en noise floor")[m
[32m+[m[32m    p.add_argument("--c3-threshold-factor", type=float, default=5.0,[m
[32m+[m[32m                   dest="c3_threshold_factor",[m
[32m+[m[32m                   help="Factor multiplicador para umbral adaptativo (default: 5.0)")[m
[32m+[m[41m    [m
[32m+[m[32m    # C3 oracle[m
[32m+[m[32m    p.add_argument("--c3-oracle", action="store_true", dest="c3_oracle",[m
[32m+[m[32m                   help="Activar oracle test (comparación forward real vs aproximado)")[m
     [m
     args = p.parse_args()[m
[32m+[m[32m    if not getattr(args, "test_mode", False) and args.run is None:[m
[32m+[m[32m        p.error("--run es requerido salvo cuando se usa --test")[m
     return Config([m
[31m-        run=args.run,[m
[32m+[m[32m        run=(args.run if args.run is not None else ("__selftest__" if getattr(args, "test_mode", False) else None)),[m
[32m+[m[32m        test_mode=getattr(args, "test_mode", False),[m
         spectrum_file=args.spectrum_file,[m
         k_features=args.k_features,[m
         cv_folds=args.cv_folds,[m
[36m@@ -123,6 +173,12 @@[m [mdef parse_args() -> Config:[m
         direct_degree=args.direct_degree,[m
         c3_metric=args.c3_metric,[m
         c3_threshold=args.c3_threshold,[m
[32m+[m[32m        c3_weights=args.c3_weights,[m
[32m+[m[32m        c3_noise_floor_eps=args.c3_noise_floor_eps,[m
[32m+[m[32m        c3_noise_floor_metric=args.c3_noise_floor_metric,[m
[32m+[m[32m        c3_adaptive_threshold=args.c3_adaptive_threshold,[m
[32m+[m[32m        c3_threshold_factor=args.c3_threshold_factor,[m
[32m+[m[32m        c3_oracle=args.c3_oracle,[m
     )[m
 [m
 [m
[36m@@ -152,6 +208,31 @@[m [mdef load_spectrum(h5_path: Path) -> dict:[m
     return data[m
 [m
 [m
[32m+[m[32mdef verify_invariants(X: np.ndarray, y: Optional[np.ndarray] = None, *, name: str = "X") -> None:[m
[32m+[m[32m    """Asserts defensivos: shapes y finitud (NaN/Inf).[m
[32m+[m
[32m+[m[32m    Objetivo: evitar debugging silencioso cuando cambie la geometría o el solver.[m
[32m+[m[32m    """[m
[32m+[m[32m    if not isinstance(X, np.ndarray):[m
[32m+[m[32m        raise TypeError(f"{name} debe ser np.ndarray, recibido {type(X)}")[m
[32m+[m[32m    if X.ndim != 2:[m
[32m+[m[32m        raise ValueError(f"{name} debe ser (n_samples, k), recibido {X.shape}")[m
[32m+[m[32m    if not np.isfinite(X).all():[m
[32m+[m[32m        bad = np.argwhere(~np.isfinite(X))[m
[32m+[m[32m        raise ValueError(f"{name} contiene NaNs/Infs en indices {bad[:5].tolist()} (mostrando hasta 5).")[m
[32m+[m
[32m+[m[32m    if y is not None:[m
[32m+[m[32m        if not isinstance(y, np.ndarray):[m
[32m+[m[32m            raise TypeError(f"y debe ser np.ndarray, recibido {type(y)}")[m
[32m+[m[32m        if y.ndim != 1:[m
[32m+[m[32m            raise ValueError(f"y debe ser (n_samples,), recibido {y.shape}")[m
[32m+[m[32m        if X.shape[0] != y.shape[0]:[m
[32m+[m[32m            raise ValueError(f"Mismatch dimensional: {name}={X.shape}, y={y.shape}")[m
[32m+[m[32m        if not np.isfinite(y).all():[m
[32m+[m[32m            bad = np.argwhere(~np.isfinite(y))[m
[32m+[m[32m            raise ValueError(f"y contiene NaNs/Infs en indices {bad[:5].tolist()} (mostrando hasta 5).")[m
[32m+[m
[32m+[m
 def compute_ratio_features(M2: np.ndarray, k: int) -> np.ndarray:[m
     """Calcula features invariantes de escala: r_n = M_n² / M_0².[m
     [m
[36m@@ -174,15 +255,17 @@[m [mdef compute_ratio_features(M2: np.ndarray, k: int) -> np.ndarray:[m
     [m
     # Ratios r_1, r_2, ..., r_k[m
     X = M2[:, 1:k+1] / M0_safe[m
[31m-    [m
[32m+[m
[32m+[m[32m    verify_invariants(X, name="X_ratios")[m
[32m+[m
     return X[m
 [m
 [m
 # =============================================================================[m
[31m-# Modelos paramétricos[m
[32m+[m[32m# Modelos paramétricos (inverso: r → Δ)[m
 # =============================================================================[m
 [m
[31m-def build_design_matrix(X: np.ndarray, model_type: str) -> np.ndarray:[m
[32m+[m[32mdef build_design_matrix(X: np.ndarray, model_type: str) -> tuple[np.ndarray, list]:[m
     """Construye matriz de diseño para el modelo especificado.[m
     [m
     Args:[m
[36m@@ -231,10 +314,12 @@[m [mdef build_design_matrix(X: np.ndarray, model_type: str) -> np.ndarray:[m
 [m
 def fit_model(Phi: np.ndarray, y: np.ndarray, ridge_lambda: float = 1e-8) -> np.ndarray:[m
     """Ajusta modelo por mínimos cuadrados (con ridge mínimo para estabilidad).[m
[31m-    [m
[32m+[m
     Returns:[m
         beta: coeficientes (n_params,)[m
     """[m
[32m+[m
[32m+[m[32m    verify_invariants(Phi, y, name="Phi")[m
     n_params = Phi.shape[1][m
     [m
     # Normal equations con ridge: (Φᵀ Φ + λI)⁻¹ Φᵀ y[m
[36m@@ -278,15 +363,9 @@[m [mdef compute_information_criteria(y: np.ndarray, y_pred: np.ndarray, n_params: in[m
     residuals = y - y_pred[m
     ss_res = np.sum(residuals ** 2)[m
     [m
[31m-    # Log-likelihood (asumiendo errores gaussianos)[m
[31m-    # L = -n/2 * log(2π) - n/2 * log(σ²) - SS_res/(2σ²)[m
[31m-    # Con σ² = SS_res/n (MLE): log(L) = -n/2 * (log(2π) + log(SS_res/n) + 1)[m
[31m-    [m
     sigma2_mle = ss_res / n[m
[31m-    log_likelihood = -n/2 * (np.log(2 * np.pi) + np.log(sigma2_mle) + 1)[m
[32m+[m[32m    log_likelihood = -n/2 * (np.log(2 * np.pi) + np.log(sigma2_mle + 1e-10) + 1)[m
     [m
[31m-    # AIC = -2 log(L) + 2k[m
[31m-    # BIC = -2 log(L) + k log(n)[m
     aic = -2 * log_likelihood + 2 * n_params[m
     bic = -2 * log_likelihood + n_params * np.log(n)[m
     [m
[36m@@ -314,21 +393,17 @@[m [mdef cross_validate(X: np.ndarray, y: np.ndarray, model_type: str,[m
     cv_predictions = np.zeros(n_samples)[m
     [m
     for fold in range(n_folds):[m
[31m-        # Índices de test[m
         start = fold * fold_size[m
         end = start + fold_size if fold < n_folds - 1 else n_samples[m
         test_idx = indices[start:end][m
         train_idx = np.concatenate([indices[:start], indices[end:]])[m
         [m
[31m-        # Split[m
         X_train, X_test = X[train_idx], X[test_idx][m
         y_train, y_test = y[train_idx], y[test_idx][m
         [m
[31m-        # Fit[m
         Phi_train, _ = build_design_matrix(X_train, model_type)[m
         beta = fit_model(Phi_train, y_train)[m
         [m
[31m-        # Predict[m
         Phi_test, _ = build_design_matrix(X_test, model_type)[m
         y_pred = predict(Phi_test, beta)[m
         [m
[36m@@ -354,36 +429,30 @@[m [mdef bootstrap_uncertainty(X: np.ndarray, y: np.ndarray, model_type: str,[m
     np.random.seed(seed)[m
     n_samples = len(y)[m
     [m
[31m-    # Ajustar modelo completo para predicciones base[m
     Phi_full, _ = build_design_matrix(X, model_type)[m
     beta_full = fit_model(Phi_full, y)[m
     y_pred_full = predict(Phi_full, beta_full)[m
     [m
[31m-    # Bootstrap: re-muestreo de filas[m
     bootstrap_predictions = np.zeros((n_bootstrap, n_samples))[m
     bootstrap_betas = [][m
     [m
     for b in range(n_bootstrap):[m
[31m-        # Muestreo con reemplazo[m
         idx = np.random.choice(n_samples, size=n_samples, replace=True)[m
         X_b, y_b = X[idx], y[idx][m
         [m
[31m-        # Fit[m
         Phi_b, _ = build_design_matrix(X_b, model_type)[m
         beta_b = fit_model(Phi_b, y_b)[m
         bootstrap_betas.append(beta_b)[m
         [m
[31m-        # Predict en datos originales[m
         y_pred_b = predict(Phi_full, beta_b)[m
         bootstrap_predictions[b] = y_pred_b[m
     [m
[31m-    # Estadísticas[m
[31m-    sigma_delta = np.std(bootstrap_predictions, axis=0)  # Por punto[m
[32m+[m[32m    sigma_delta = np.std(bootstrap_predictions, axis=0)[m
     beta_mean = np.mean(bootstrap_betas, axis=0)[m
     beta_std = np.std(bootstrap_betas, axis=0)[m
     [m
     return {[m
[31m-        "sigma_delta": sigma_delta,                    # (n_samples,)[m
[32m+[m[32m        "sigma_delta": sigma_delta,[m
         "sigma_delta_mean": float(np.mean(sigma_delta)),[m
         "sigma_delta_max": float(np.max(sigma_delta)),[m
         "beta_mean": beta_mean,[m
[36m@@ -402,12 +471,10 @@[m [mdef select_best_model(X: np.ndarray, y: np.ndarray, models: tuple,[m
     results = {}[m
     [m
     for model_type in models:[m
[31m-        # Fit completo[m
         Phi, feature_names = build_design_matrix(X, model_type)[m
         beta = fit_model(Phi, y)[m
         y_pred = predict(Phi, beta)[m
         [m
[31m-        # Métricas[m
         metrics = compute_metrics(y, y_pred)[m
         info_criteria = compute_information_criteria(y, y_pred, len(beta))[m
         cv_results = cross_validate(X, y, model_type, cv_folds, seed)[m
[36m@@ -421,7 +488,6 @@[m [mdef select_best_model(X: np.ndarray, y: np.ndarray, models: tuple,[m
             "cv": cv_results,[m
         }[m
     [m
[31m-    # Seleccionar por BIC (menor es mejor)[m
     best_model = min(results.keys(), key=lambda m: results[m]["info_criteria"]["bic"])[m
     [m
     return {[m
[36m@@ -432,32 +498,21 @@[m [mdef select_best_model(X: np.ndarray, y: np.ndarray, models: tuple,[m
 [m
 [m
 # =============================================================================[m
[31m-# Contratos[m
[32m+[m[32m# Contratos C1 y C2[m
 # =============================================================================[m
 [m
 def evaluate_c1_ising(delta_pred: np.ndarray, sigma_delta: np.ndarray,[m
                       delta_target: float, tau: float, sigma_factor: float,[m
                       target_name: str, sigma_cap: float = 0.1,[m
                       delta_range: tuple[float, float] = None) -> dict:[m
[31m-    """Contrato C1: compatibilidad puntual con Ising.[m
[31m-    [m
[31m-    Criterios:[m
[31m-    - OUT_OF_DOMAIN: target fuera del rango [delta_min, delta_max] del dataset[m
[31m-    - STRONG_PASS: |Δ̂ - Δ*| ≤ τ (tolerancia absoluta)[m
[31m-    - WEAK_PASS: |Δ̂ - Δ*| ≤ 2σ AND σ ≤ sigma_cap (incertidumbre acotada)[m
[31m-    - FAIL: ninguno de los anteriores[m
[31m-    [m
[31m-    El sigma_cap evita "pasar por incertidumbre gigante".[m
[31m-    """[m
[32m+[m[32m    """Contrato C1: compatibilidad puntual con Ising."""[m
     errors = np.abs(delta_pred - delta_target)[m
     [m
[31m-    # Verificar si el target está en dominio[m
     in_domain = True[m
     if delta_range is not None:[m
         delta_min, delta_max = delta_range[m
         in_domain = delta_min <= delta_target <= delta_max[m
     [m
[31m-    # Si está fuera de dominio, no evaluamos pass/fail[m
     if not in_domain:[m
         best_idx = np.argmin(errors)[m
         return {[m
[36m@@ -482,22 +537,15 @@[m [mdef evaluate_c1_ising(delta_pred: np.ndarray, sigma_delta: np.ndarray,[m
             "n_sigma_too_large": 0,[m
         }[m
     [m
[31m-    # Criterio fuerte: tolerancia absoluta[m
     pass_tau = errors <= tau[m
[31m-    [m
[31m-    # Criterio débil: dentro de 2σ, pero solo si σ es razonable[m
     sigma_usable = sigma_delta <= sigma_cap[m
     pass_sigma_raw = errors <= sigma_factor * sigma_delta[m
     pass_sigma = pass_sigma_raw & sigma_usable[m
     [m
[31m-    # Mejor candidato: primero por pass_tau, luego por pass_sigma, luego por error mínimo[m
[31m-    # Construir score: 0 si pass_tau, 1 si pass_sigma, 2 si ninguno, luego desempatar por error[m
     scores = np.where(pass_tau, 0, np.where(pass_sigma, 1, 2))[m
[31m-    # Combinar score con error normalizado para desempate[m
     combined_score = scores + errors / (errors.max() + 1e-10)[m
     best_idx = np.argmin(combined_score)[m
     [m
[31m-    # Status del mejor candidato[m
     if pass_tau[best_idx]:[m
         status = "STRONG_PASS"[m
     elif pass_sigma[best_idx]:[m
[36m@@ -507,7 +555,6 @@[m [mdef evaluate_c1_ising(delta_pred: np.ndarray, sigma_delta: np.ndarray,[m
     else:[m
         status = "FAIL"[m
     [m
[31m-    # Status global[m
     if np.any(pass_tau):[m
         global_status = "STRONG_PASS"[m
     elif np.any(pass_sigma):[m
[36m@@ -544,8 +591,6 @@[m [mdef evaluate_c2_consistency(y_true: np.ndarray, y_pred: np.ndarray,[m
                             cv_rmse: float) -> dict:[m
     """Contrato C2: consistencia interna del diccionario."""[m
     metrics = compute_metrics(y_true, y_pred)[m
[31m-    [m
[31m-    # Criterio de fallo: CV RMSE > 0.05[m
     consistency_ok = cv_rmse < 0.05[m
     [m
     return {[m
[36m@@ -558,54 +603,53 @@[m [mdef evaluate_c2_consistency(y_true: np.ndarray, y_pred: np.ndarray,[m
     }[m
 [m
 [m
[31m-def fit_direct_model(y: np.ndarray, X: np.ndarray, model_type: str, degree: int = 2) -> tuple[np.ndarray, list]:[m
[31m-    """Ajusta modelo directo Delta -> ratios (para C3).[m
[32m+[m[32m# =============================================================================[m
[32m+[m[32m# Contrato C3: Modelo directo y métricas[m
[32m+[m[32m# =============================================================================[m
[32m+[m
[32m+[m[32m# =============================================================================[m
[32m+[m[32m# Modelo Directo (Proxy para C3)[m
[32m+[m[32m# NOTA: Este modelo polinomial NO es la física real. Es un proxy suave local[m
[32m+[m[32m# usado para evaluar la invertibilidad del mapeo.[m
[32m+[m[32m# Si C3a falla, significa que el proxy es malo, no que la física esté rota.[m
[32m+[m[32m# =============================================================================[m
[32m+[m
[32m+[m
[32m+[m[32mdef build_direct_design_matrix(y: np.ndarray, degree: int) -> np.ndarray:[m
[32m+[m[32m    """Construye matriz de diseño polinomial para modelo directo Δ→r."""[m
[32m+[m[32m    n_samples = len(y)[m
[32m+[m[32m    # [1, Δ, Δ², ..., Δ^degree][m
[32m+[m[32m    Phi = np.column_stack([y**p for p in range(degree + 1)])[m
[32m+[m[32m    return Phi[m
[32m+[m
[32m+[m
[32m+[m[32mdef fit_direct_model(y: np.ndarray, X: np.ndarray, degree: int) -> list[np.ndarray]:[m
[32m+[m[32m    """Ajusta modelo directo Δ → ratios.[m
     [m
     Args:[m
[31m-        y: array (n_samples,) con Delta[m
[32m+[m[32m        y: array (n_samples,) con Δ[m
         X: array (n_samples, k) con ratios objetivo[m
[31m-        model_type: "linear" o "poly"[m
[31m-        degree: grado del polinomio si model_type="poly"[m
[32m+[m[32m        degree: grado del polinomio[m
     [m
     Returns:[m
         betas: lista de coeficientes, uno por ratio[m
[31m-        feature_names: nombres de features del modelo[m
     """[m
     n_samples, k = X.shape[m
[31m-    betas = [][m
[32m+[m[32m    Phi = build_direct_design_matrix(y, degree)[m
     [m
[31m-    # Para el modelo directo, las features son funciones de Delta[m
[31m-    if model_type == "linear":[m
[31m-        Phi = np.column_stack([np.ones(n_samples), y])[m
[31m-        names = ["1", "Delta"][m
[31m-    elif model_type == "poly":[m
[31m-        # [1, Delta, Delta^2, ..., Delta^degree][m
[31m-        cols = [np.ones(n_samples)] + [y**p for p in range(1, degree + 1)][m
[31m-        Phi = np.column_stack(cols)[m
[31m-        names = ["1"] + [f"Delta^{p}" for p in range(1, degree + 1)][m
[31m-    else:[m
[31m-        raise ValueError(f"model_type desconocido: {model_type}")[m
[31m-    [m
[31m-    # Ajustar un modelo por cada ratio[m
[32m+[m[32m    betas = [][m
     for j in range(k):[m
         beta_j = fit_model(Phi, X[:, j])[m
         betas.append(beta_j)[m
     [m
[31m-    return betas, names[m
[32m+[m[32m    return betas[m
 [m
 [m
[31m-def predict_ratios(y: np.ndarray, betas: list, model_type: str, degree: int = 2) -> np.ndarray:[m
[31m-    """Predice ratios desde Delta usando modelo directo."""[m
[32m+[m[32mdef predict_ratios(y: np.ndarray, betas: list[np.ndarray], degree: int) -> np.ndarray:[m
[32m+[m[32m    """Predice ratios desde Δ usando modelo directo."""[m
     n_samples = len(y)[m
     k = len(betas)[m
[31m-    [m
[31m-    if model_type == "linear":[m
[31m-        Phi = np.column_stack([np.ones(n_samples), y])[m
[31m-    elif model_type == "poly":[m
[31m-        cols = [np.ones(n_samples)] + [y**p for p in range(1, degree + 1)][m
[31m-        Phi = np.column_stack(cols)[m
[31m-    else:[m
[31m-        raise ValueError(f"model_type desconocido: {model_type}")[m
[32m+[m[32m    Phi = build_direct_design_matrix(y, degree)[m
     [m
     X_pred = np.zeros((n_samples, k))[m
     for j, beta_j in enumerate(betas):[m
[36m@@ -614,99 +658,385 @@[m [mdef predict_ratios(y: np.ndarray, betas: list, model_type: str, degree: int = 2)[m
     return X_pred[m
 [m
 [m
[31m-def evaluate_c3_spectral(X: np.ndarray, X_pred: np.ndarray,[m
[31m-                         metric: str = "rmse", threshold: float = 0.05,[m
[31m-                         is_placeholder: bool = False) -> dict:[m
[31m-    """Contrato C3: compatibilidad espectral (error en ratios).[m
[32m+[m[32mdef compute_weights(k: int, scheme: str, X: np.ndarray = None) -> np.ndarray:[m
[32m+[m[32m    """Calcula pesos para métricas C3.[m
     [m
     Args:[m
[31m-        X: ratios originales[m
[31m-        X_pred: ratios reconstruidos[m
[31m-        metric: "rmse" | "rmse_log" | "rmse_rel"[m
[31m-        threshold: umbral para PASS/FAIL[m
[31m-        is_placeholder: si True, retorna SKIP[m
[32m+[m[32m        k: número de ratios[m
[32m+[m[32m        scheme: "none", "inv_n", "inv_n2", "inv_n4", "inv_r2"[m
[32m+[m[32m        X: array (n_samples, k) de ratios (requerido para inv_r2)[m
     [m
[31m-    Si is_placeholder=True, marca como SKIP en lugar de evaluar.[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        weights: array (k,) con pesos normalizados[m
     """[m
[31m-    if is_placeholder:[m
[31m-        return {[m
[31m-            "status": "SKIP",[m
[31m-            "rmse_per_ratio": [],[m
[31m-            "rmse_global": None,[m
[31m-            "spectral_ok": None,[m
[31m-            "metric": metric,[m
[31m-            "threshold": threshold,[m
[31m-            "note": "C3 requiere modelo directo. Use --enable-c3 para activar.",[m
[31m-        }[m
[32m+[m[32m    n = np.arange(1, k + 1)  # n = 1, 2, ..., k[m
[32m+[m[41m    [m
[32m+[m[32m    if scheme == "none":[m
[32m+[m[32m        w = np.ones(k)[m
[32m+[m[32m    elif scheme == "inv_n":[m
[32m+[m[32m        w = 1.0 / n[m
[32m+[m[32m    elif scheme == "inv_n2":[m
[32m+[m[32m        w = 1.0 / (n ** 2)[m
[32m+[m[32m    elif scheme == "inv_n4":[m
[32m+[m[32m        w = 1.0 / (n ** 4)[m
[32m+[m[32m    elif scheme == "inv_r2":[m
[32m+[m[32m        if X is None:[m
[32m+[m[32m            raise ValueError("inv_r2 requiere X (ratios)")[m
[32m+[m[32m        # Media de r_n² sobre muestras + epsilon para estabilidad[m
[32m+[m[32m        r_mean_sq = np.mean(X ** 2, axis=0) + 1e-12[m
[32m+[m[32m        w = 1.0 / r_mean_sq[m
[32m+[m[32m    else:[m
[32m+[m[32m        raise ValueError(f"Esquema de pesos desconocido: {scheme}")[m
[32m+[m[41m    [m
[32m+[m[32m    # Normalizar para que sumen 1[m
[32m+[m[32m    w = w / np.sum(w)[m
[32m+[m[32m    return w[m
[32m+[m
[32m+[m
[32m+[m[32mdef compute_ratio_distance([m
[32m+[m[32m    r_true: np.ndarray,[m[41m [m
[32m+[m[32m    r_pred: np.ndarray,[m[41m [m
[32m+[m[32m    metric: str,[m
[32m+[m[32m    weights: np.ndarray = None[m
[32m+[m[32m) -> dict:[m
[32m+[m[32m    """Función unificada para calcular distancia entre ratios.[m
[32m+[m[41m    [m
[32m+[m[32m    Args:[m
[32m+[m[32m        r_true: array (n_samples, k) ratios verdaderos[m
[32m+[m[32m        r_pred: array (n_samples, k) ratios predichos[m
[32m+[m[32m        metric: "rmse", "rmse_log", "rmse_rel"[m
[32m+[m[32m        weights: array (k,) pesos por ratio (opcional)[m
[32m+[m[41m    [m
[32m+[m[32m    Returns:[m
[32m+[m[32m        dict con global, per_ratio, weights_used[m
[32m+[m[32m    """[m
[32m+[m[32m    n_samples, k = r_true.shape[m
     [m
[31m-    eps = 1e-12[m
[32m+[m[32m    if weights is None:[m
[32m+[m[32m        weights = np.ones(k) / k  # Pesos uniformes[m
     [m
[31m-    # Calcular errores segun metrica[m
[32m+[m[32m    # Calcular errores según métrica[m
     if metric == "rmse":[m
[31m-        errors = X - X_pred[m
[32m+[m[32m        errors = r_true - r_pred  # (n_samples, k)[m
[32m+[m[32m        errors_sq = errors ** 2[m
     elif metric == "rmse_log":[m
[31m-        # RMSE sobre log(r) - estabiliza ratios grandes[m
[31m-        X_safe = np.clip(X, eps, None)[m
[31m-        X_pred_safe = np.clip(X_pred, eps, None)[m
[31m-        errors = np.log(X_safe) - np.log(X_pred_safe)[m
[32m+[m[32m        # log(r) con clamp para evitar log(0)[m
[32m+[m[32m        r_true_safe = np.maximum(r_true, 1e-10)[m
[32m+[m[32m        r_pred_safe = np.maximum(r_pred, 1e-10)[m
[32m+[m[32m        errors = np.log(r_true_safe) - np.log(r_pred_safe)[m
[32m+[m[32m        errors_sq = errors ** 2[m
     elif metric == "rmse_rel":[m
[31m-        # RMSE relativo[m
[31m-        denom = np.clip(np.abs(X), eps, None)[m
[31m-        errors = (X - X_pred) / denom[m
[32m+[m[32m        # Error relativo: (r - r̂) / r[m
[32m+[m[32m        r_true_safe = np.where(np.abs(r_true) > 1e-10, r_true, 1e-10)[m
[32m+[m[32m        errors = (r_true - r_pred) / r_true_safe[m
[32m+[m[32m        errors_sq = errors ** 2[m
     else:[m
[31m-        raise ValueError(f"metric desconocida: {metric}")[m
[32m+[m[32m        raise ValueError(f"Métrica desconocida: {metric}")[m
     [m
[31m-    rmse_per_ratio = np.sqrt(np.mean(errors ** 2, axis=0))[m
[31m-    rmse_global = np.sqrt(np.mean(errors ** 2))[m
[31m-    spectral_ok = bool(rmse_global < threshold)[m
[32m+[m[32m    # MSE por ratio (promedio sobre muestras)[m
[32m+[m[32m    mse_per_ratio = np.mean(errors_sq, axis=0)  # (k,)[m
[32m+[m[32m    rmse_per_ratio = np.sqrt(mse_per_ratio)[m
[32m+[m[41m    [m
[32m+[m[32m    # WRMSE global: sqrt(sum_n w_n * mse_n)[m
[32m+[m[32m    wrmse_global = np.sqrt(np.sum(weights * mse_per_ratio))[m
     [m
     return {[m
[31m-        "status": "PASS" if spectral_ok else "FAIL",[m
[32m+[m[32m        "global": float(wrmse_global),[m
[32m+[m[32m        "per_ratio": [float(r) for r in rmse_per_ratio],[m
[32m+[m[32m        "weights_used": [float(w) for w in weights],[m
         "metric": metric,[m
[31m-        "rmse_per_ratio": [float(r) for r in rmse_per_ratio],[m
[31m-        "rmse_global": float(rmse_global),[m
[31m-        "threshold": threshold,[m
[31m-        "spectral_ok": spectral_ok,[m
     }[m
 [m
 [m
[31m-def evaluate_c3_cycle(X: np.ndarray, y: np.ndarray, [m
[31m-                      inverse_model_type: str,[m
[31m-                      direct_model_type: str = "poly",[m
[31m-                      direct_degree: int = 2,[m
[31m-                      c3_metric: str = "rmse",[m
[31m-                      c3_threshold: float = 0.05) -> dict:[m
[31m-    """Contrato C3 completo: test de ciclo r -> Delta_hat -> r_hat.[m
[32m+[m[32mdef compute_sensitivity([m
[32m+[m[32m    y: np.ndarray,[m
[32m+[m[32m    betas_direct: list[np.ndarray],[m
[32m+[m[32m    degree: int,[m
[32m+[m[32m    eps: float = 0.001[m
[32m+[m[32m) -> dict:[m
[32m+[m[32m    """Calcula sensibilidad del modelo directo: |r(Δ+ε) - r(Δ)| / ε.[m
[32m+[m[41m    [m
[32m+[m[32m    Proxy de condicionamiento del observable.[m
[32m+[m[32m    """[m
[32m+[m[32m    n_samples = len(y)[m
[32m+[m[32m    k = len(betas_direct)[m
     [m
[31m-    1. Ajusta modelo inverso: r -> Delta[m
[31m-    2. Ajusta modelo directo: Delta -> r[m
[31m-    3. Evalua ciclo: r -> Delta_hat -> r_hat[m
[31m-    4. Mide error(r, r_hat) segun metrica[m
[32m+[m[32m    # r(Δ) y r(Δ+ε)[m
[32m+[m[32m    r_base = predict_ratios(y, betas_direct, degree)[m
[32m+[m[32m    r_perturbed = predict_ratios(y + eps, betas_direct, degree)[m
     [m
[31m-    Args:[m
[31m-        X: ratios originales[m
[31m-        y: Delta (target del modelo inverso)[m
[31m-        inverse_model_type: tipo de modelo inverso ("linear", "poly2")[m
[31m-        direct_model_type: tipo de modelo directo ("linear", "poly")[m
[31m-        direct_degree: grado del polinomio para modelo directo[m
[31m-        c3_metric: metrica de error ("rmse", "rmse_log", "rmse_rel")[m
[31m-        c3_threshold: umbral para PASS/FAIL[m
[32m+[m[32m    # Sensibilidad por componente: |dr/dΔ| ≈ |r(Δ+ε) - r(Δ)| / ε[m
[32m+[m[32m    sensitivity = np.abs(r_perturbed - r_base) / eps  # (n_samples, k)[m
[32m+[m[41m    [m
[32m+[m[32m    # Agregar estadísticas[m
[32m+[m[32m    sensitivity_median_per_ratio = np.median(sensitivity, axis=0)  # (k,)[m
[32m+[m[41m    [m
[32m+[m[32m    # Norma L2 de sensibilidad por muestra, luego mediana[m
[32m+[m[32m    sensitivity_norm_per_sample = np.linalg.norm(sensitivity, axis=1)  # (n_samples,)[m
[32m+[m[32m    sensitivity_norm_median = float(np.median(sensitivity_norm_per_sample))[m
[32m+[m[41m    [m
[32m+[m[32m    return {[m
[32m+[m[32m        "eps": eps,[m
[32m+[m[32m        "per_ratio_median": [float(s) for s in sensitivity_median_per_ratio],[m
[32m+[m[32m        "norm_median": sensitivity_norm_median,[m
[32m+[m[32m        "norm_max": float(np.max(sensitivity_norm_per_sample)),[m
[32m+[m[32m        "norm_p90": float(np.percentile(sensitivity_norm_per_sample, 90)),[m
[32m+[m[32m    }[m
[32m+[m
[32m+[m
[32m+[m[32mdef compute_noise_floor([m
[32m+[m[32m    y: np.ndarray,[m
[32m+[m[32m    X: np.ndarray,[m
[32m+[m[32m    betas_direct: list[np.ndarray],[m
[32m+[m[32m    degree: int,[m
[32m+[m[32m    eps: float,[m
[32m+[m[32m    metric: str,[m
[32m+[m[32m    weights: np.ndarray[m
[32m+[m[32m) -> dict:[m
[32m+[m[32m    """Estima noise floor: resolución efectiva del observable bajo perturbación.[m
[32m+[m[41m    [m
[32m+[m[32m    Mide cuánto cambia r cuando Δ cambia por ±ε.[m
     """[m
[31m-    # Modelo inverso (ya lo tenemos, pero lo re-ajustamos para claridad)[m
[31m-    Phi_inv, _ = build_design_matrix(X, inverse_model_type)[m
[31m-    beta_inv = fit_model(Phi_inv, y)[m
[32m+[m[32m    n_samples = len(y)[m
[32m+[m[41m    [m
[32m+[m[32m    # r predicho en Δ y en Δ+ε[m
[32m+[m[32m    r_base = predict_ratios(y, betas_direct, degree)[m
[32m+[m[32m    r_perturbed = predict_ratios(y + eps, betas_direct, degree)[m
[32m+[m[41m    [m
[32m+[m[32m    # Calcular "error" de la perturbación usando la métrica elegida[m
[32m+[m[32m    distance = compute_ratio_distance(r_base, r_perturbed, metric, weights)[m
[32m+[m[41m    [m
[32m+[m[32m    # También calcular por muestra para estadísticas[m
[32m+[m[32m    sigmas = [][m
[32m+[m[32m    for i in range(n_samples):[m
[32m+[m[32m        sample_dist = compute_ratio_distance([m
[32m+[m[32m            r_base[i:i+1], r_perturbed[i:i+1], metric, weights[m
[32m+[m[32m        )[m
[32m+[m[32m        sigmas.append(sample_dist["global"])[m
[32m+[m[41m    [m
[32m+[m[32m    sigmas = np.array(sigmas)[m
[32m+[m[41m    [m
[32m+[m[32m    return {[m
[32m+[m[32m        "eps": eps,[m
[32m+[m[32m        "metric": metric,[m
[32m+[m[32m        "median_sigma": float(np.median(sigmas)),[m
[32m+[m[32m        "mean_sigma": float(np.mean(sigmas)),[m
[32m+[m[32m        "p90_sigma": float(np.percentile(sigmas, 90)),[m
[32m+[m[32m        "max_sigma": float(np.max(sigmas)),[m
[32m+[m[32m        "aggregate_distance": distance["global"],[m
[32m+[m[32m    }[m
[32m+[m
[32m+[m
[32m+[m[32mdef evaluate_c3a_decoder([m
[32m+[m[32m    X_true: np.ndarray,[m
[32m+[m[32m    y_true: np.ndarray,[m
[32m+[m[32m    betas_direct: list[np.ndarray],[m
[32m+[m[32m    degree: int,[m
[32m+[m[32m    metric: str,[m
[32m+[m[32m    weights: np.ndarray[m
[32m+[m[32m) -> dict:[m
[32m+[m[32m    """C3a: Evalúa modelo directo (decoder) Δ_true → r̂.[m
[32m+[m[41m    [m
[32m+[m[32m    Compara r_true con r̂ = decoder(Δ_true).[m
[32m+[m[32m    """[m
[32m+[m[32m    r_pred = predict_ratios(y_true, betas_direct, degree)[m
[32m+[m[41m    [m
[32m+[m[32m    # Calcular todas las métricas[m
[32m+[m[32m    result = {}[m
[32m+[m[32m    for m in ["rmse", "rmse_log", "rmse_rel"]:[m
[32m+[m[32m        dist = compute_ratio_distance(X_true, r_pred, m, weights if m == metric else None)[m
[32m+[m[32m        result[m] = dist["global"][m
[32m+[m[32m        if m == metric:[m
[32m+[m[32m            result["per_ratio"] = dist["per_ratio"][m
[32m+[m[41m    [m
[32m+[m[32m    result["metric_primary"] = metric[m
[32m+[m[32m    result["global"] = result[metric][m
[32m+[m[41m    [m
[32m+[m[32m    return result[m
[32m+[m
[32m+[m
[32m+[m[32mdef evaluate_c3b_cycle([m
[32m+[m[32m    X_true: np.ndarray,[m
[32m+[m[32m    y_true: np.ndarray,[m
[32m+[m[32m    inverse_model_type: str,[m
[32m+[m[32m    betas_direct: list[np.ndarray],[m
[32m+[m[32m    degree: int,[m
[32m+[m[32m    metric: str,[m
[32m+[m[32m    weights: np.ndarray[m
[32m+[m[32m) -> dict:[m
[32m+[m[32m    """C3b: Evalúa ciclo completo r → Δ̂ → r̂.[m
[32m+[m[41m    [m
[32m+[m[32m    1. Ajusta modelo inverso r → Δ[m
[32m+[m[32m    2. Predice Δ̂ desde r_true[m
[32m+[m[32m    3. Predice r̂ = decoder(Δ̂)[m
[32m+[m[32m    4. Compara r_true vs r̂[m
[32m+[m[32m    """[m
[32m+[m[32m    # Modelo inverso[m
[32m+[m[32m    Phi_inv, _ = build_design_matrix(X_true, inverse_model_type)[m
[32m+[m[32m    beta_inv = fit_model(Phi_inv, y_true)[m
     delta_pred = predict(Phi_inv, beta_inv)[m
     [m
[31m-    # Modelo directo[m
[31m-    betas_dir, _ = fit_direct_model(y, X, direct_model_type, direct_degree)[m
[32m+[m[32m    # Ciclo: r → Δ̂ → r̂[m
[32m+[m[32m    r_reconstructed = predict_ratios(delta_pred, betas_direct, degree)[m
[32m+[m[41m    [m
[32m+[m[32m    # Calcular todas las métricas[m
[32m+[m[32m    result = {}[m
[32m+[m[32m    for m in ["rmse", "rmse_log", "rmse_rel"]:[m
[32m+[m[32m        dist = compute_ratio_distance(X_true, r_reconstructed, m, weights if m == metric else None)[m
[32m+[m[32m        result[m] = dist["global"][m
[32m+[m[32m        if m == metric:[m
[32m+[m[32m            result["per_ratio"] = dist["per_ratio"][m
[32m+[m[41m    [m
[32m+[m[32m    result["metric_primary"] = metric[m
[32m+[m[32m    result["global"] = result[metric][m
[32m+[m[32m    result["delta_pred_range"] = [float(delta_pred.min()), float(delta_pred.max())][m
[32m+[m[41m    [m
[32m+[m[32m    return result[m
[32m+[m
[32m+[m
[32m+[m[32mdef evaluate_c3_oracle([m
[32m+[m[32m    X_true: np.ndarray,[m
[32m+[m[32m    y_true: np.ndarray,[m
[32m+[m[32m    betas_direct: list[np.ndarray],[m
[32m+[m[32m    degree: int,[m
[32m+[m[32m    metric: str,[m
[32m+[m[32m    weights: np.ndarray[m
[32m+[m[32m) -> dict:[m
[32m+[m[32m    """Oracle test: compara forward "real" (datos) vs forward aproximado (decoder).[m
[32m+[m[41m    [m
[32m+[m[32m    En baseline, esto es equivalente a C3a. Pero se etiqueta explícitamente[m
[32m+[m[32m    para hacer el gap entre forward real y aproximado defendible.[m
[32m+[m[32m    """[m
[32m+[m[32m    # En baseline, el "forward real" son los ratios del dataset[m
[32m+[m[32m    # El "forward aproximado" es el decoder[m
[32m+[m[32m    # Esto es idéntico a C3a, pero con etiqueta explícita[m
[32m+[m[41m    [m
[32m+[m[32m    r_pred = predict_ratios(y_true, betas_direct, degree)[m
[32m+[m[41m    [m
[32m+[m[32m    result = {}[m
[32m+[m[32m    for m in ["rmse", "rmse_log", "rmse_rel"]:[m
[32m+[m[32m        dist = compute_ratio_distance(X_true, r_pred, m, weights if m == metric else None)[m
[32m+[m[32m        result[m] = dist["global"][m
[32m+[m[32m        if m == metric:[m
[32m+[m[32m            result["per_ratio"] = dist["per_ratio"][m
[32m+[m[41m    [m
[32m+[m[32m    result["metric_primary"] = metric[m
[32m+[m[32m    result["global"] = result[metric][m
[32m+[m[32m    result["description"] = "Gap entre forward real (datos) y forward aproximado (decoder)"[m
[32m+[m[41m    [m
[32m+[m[32m    return result[m
[32m+[m
[32m+[m
[32m+[m[32mdef evaluate_c3_full([m
[32m+[m[32m    X: np.ndarray,[m
[32m+[m[32m    y: np.ndarray,[m
[32m+[m[32m    cfg: Config,[m
[32m+[m[32m    inverse_model_type: str[m
[32m+[m[32m) -> dict:[m
[32m+[m[32m    """Evaluación completa de C3 con diagnóstico causal."""[m
[32m+[m
[32m+[m[32m    verify_invariants(X, y, name="X")[m
[32m+[m
[32m+[m[32m    if not cfg.enable_c3:[m
[32m+[m[32m        return {[m
[32m+[m[32m            "status": "SKIP",[m
[32m+[m[32m            "failure_mode": None,[m
[32m+[m[32m            "note": "C3 requiere --enable-c3. Usar --enable-c3 para activar.",[m
[32m+[m[32m        }[m
[32m+[m[41m    [m
[32m+[m[32m    k = cfg.k_features[m
[32m+[m[32m    metric = cfg.c3_metric[m
[32m+[m[32m    noise_floor_metric = cfg.c3_noise_floor_metric or metric[m
[32m+[m[41m    [m
[32m+[m[32m    # Invariantes defensivos[m
[32m+[m[32m    verify_invariants(X, y, name="X_ratios")[m
[32m+[m
[32m+[m[32m    # Calcular pesos[m
[32m+[m[32m    weights = compute_weights(k, cfg.c3_weights, X)[m
[32m+[m[41m    [m
[32m+[m[32m    # Ajustar modelo directo[m
[32m+[m[32m    betas_direct = fit_direct_model(y, X, cfg.direct_degree)[m
[32m+[m[41m    [m
[32m+[m[32m    # C3a: Decoder[m
[32m+[m[32m    c3a = evaluate_c3a_decoder(X, y, betas_direct, cfg.direct_degree, metric, weights)[m
[32m+[m[41m    [m
[32m+[m[32m    # C3b: Cycle[m
[32m+[m[32m    c3b = evaluate_c3b_cycle(X, y, inverse_model_type, betas_direct, cfg.direct_degree, metric, weights)[m
[32m+[m[41m    [m
[32m+[m[32m    # Sensitivity[m
[32m+[m[32m    sensitivity = compute_sensitivity(y, betas_direct, cfg.direct_degree, cfg.c3_noise_floor_eps)[m
[32m+[m[41m    [m
[32m+[m[32m    # Noise floor[m
[32m+[m[32m    noise_floor = compute_noise_floor([m
[32m+[m[32m        y, X, betas_direct, cfg.direct_degree,[m
[32m+[m[32m        cfg.c3_noise_floor_eps, noise_floor_metric, weights[m
[32m+[m[32m    )[m
[32m+[m[41m    [m
[32m+[m[32m    # Oracle (opcional)[m
[32m+[m[32m    oracle = None[m
[32m+[m[32m    if cfg.c3_oracle:[m
[32m+[m[32m        oracle = evaluate_c3_oracle(X, y, betas_direct, cfg.direct_degree, metric, weights)[m
[32m+[m[41m    [m
[32m+[m[32m    # Calcular umbral efectivo[m
[32m+[m[32m    threshold_user = cfg.c3_threshold[m
[32m+[m[32m    threshold_adaptive = cfg.c3_threshold_factor * noise_floor["median_sigma"][m
     [m
[31m-    # Ciclo: r -> Delta_hat -> r_hat[m
[31m-    X_reconstructed = predict_ratios(delta_pred, betas_dir, direct_model_type, direct_degree)[m
[32m+[m[32m    if cfg.c3_adaptive_threshold:[m
[32m+[m[32m        threshold_effective = max(threshold_user, threshold_adaptive)[m
[32m+[m[32m        threshold_mode = "adaptive"[m
[32m+[m[32m    else:[m
[32m+[m[32m        threshold_effective = threshold_user[m
[32m+[m[32m        threshold_mode = "user"[m
[32m+[m[41m    [m
[32m+[m[32m    # Determinar failure_mode y status[m
[32m+[m[32m    c3a_value = c3a["global"][m
[32m+[m[32m    c3b_value = c3b["global"][m
[32m+[m
[32m+[m[32m    c3a_ok = bool(c3a_value <= threshold_effective)[m
[32m+[m[32m    c3b_ok = bool(c3b_value <= threshold_effective)[m
[32m+[m[32m    c3a_status = "PASS" if c3a_ok else "FAIL"[m
[32m+[m[32m    c3b_status = "PASS" if c3b_ok else "FAIL"[m
     [m
[31m-    # Metricas del ciclo[m
[31m-    return evaluate_c3_spectral(X, X_reconstructed, [m
[31m-                                metric=c3_metric, threshold=c3_threshold,[m
[31m-                                is_placeholder=False)[m
[32m+[m[32m    if c3a_value > threshold_effective:[m
[32m+[m[32m        failure_mode = "DECODER_MISMATCH"[m
[32m+[m[32m        status = "FAIL"[m
[32m+[m[32m    elif c3b_value > threshold_effective:[m
[32m+[m[32m        failure_mode = "CYCLE_INCONSISTENT"[m
[32m+[m[32m        status = "FAIL"[m
[32m+[m[32m    else:[m
[32m+[m[32m        failure_mode = None[m
[32m+[m[32m        status = "PASS"[m
[32m+[m[41m    [m
[32m+[m[32m    return {[m
[32m+[m[32m        "status": status,[m
[32m+[m[32m        "c3a_status": c3a_status,[m
[32m+[m[32m        "c3b_status": c3b_status,[m
[32m+[m[32m        "failure_mode": failure_mode,[m
[32m+[m[32m        "metric": metric,[m
[32m+[m[32m        "weights": {[m
[32m+[m[32m            "scheme": cfg.c3_weights,[m
[32m+[m[32m            "values": [float(w) for w in weights],[m
[32m+[m[32m        },[m
[32m+[m[32m        "c3a_decoder": c3a,[m
[32m+[m[32m        "c3b_cycle": c3b,[m
[32m+[m[32m        "sensitivity": sensitivity,[m
[32m+[m[32m        "noise_floor": noise_floor,[m
[32m+[m[32m        "threshold": {[m
[32m+[m[32m            "user": threshold_user,[m
[32m+[m[32m            "adaptive": float(threshold_adaptive),[m
[32m+[m[32m            "effective": float(threshold_effective),[m
[32m+[m[32m            "factor": cfg.c3_threshold_factor,[m
[32m+[m[32m            "mode": threshold_mode,[m
[32m+[m[32m        },[m
[32m+[m[32m        "oracle": oracle,[m
[32m+[m[32m        "direct_model": {[m
[32m+[m[32m            "type": cfg.direct_model,[m
[32m+[m[32m            "degree": cfg.direct_degree,[m
[32m+[m[32m        },[m
[32m+[m[32m        # Backward compatibility[m
[32m+[m[32m        "spectral_ok": status == "PASS",[m
[32m+[m[32m        "rmse_global": c3b_value,  # Para compatibilidad con código anterior[m
[32m+[m[32m    }[m
 [m
 [m
 # =============================================================================[m
[36m@@ -714,17 +1044,10 @@[m [mdef evaluate_c3_cycle(X: np.ndarray, y: np.ndarray,[m
 # =============================================================================[m
 [m
 def build_atlas(delta: np.ndarray, M2: np.ndarray, k: int) -> dict:[m
[31m-    """Construye atlas simple de teorías efectivas.[m
[31m-    [m
[31m-    En hard-wall es trivial (cada Δ define una teoría), pero dejamos[m
[31m-    el formato preparado para geometrías más complejas.[m
[31m-    """[m
[32m+[m[32m    """Construye atlas simple de teorías efectivas."""[m
     X = compute_ratio_features(M2, k)[m
[31m-    [m
[31m-    # Clustering simple por Δ (ya están ordenados)[m
     n_theories = len(delta)[m
     [m
[31m-    # Estadísticas por teoría[m
     theories = [][m
     for i in range(n_theories):[m
         theories.append({[m
[36m@@ -739,7 +1062,7 @@[m [mdef build_atlas(delta: np.ndarray, M2: np.ndarray, k: int) -> dict:[m
         "n_theories": n_theories,[m
         "delta_range": [float(delta.min()), float(delta.max())],[m
         "theories": theories,[m
[31m-        "clustering_method": "parametric_sweep",  # Trivial para hard-wall[m
[32m+[m[32m        "clustering_method": "parametric_sweep",[m
     }[m
 [m
 [m
[36m@@ -771,14 +1094,12 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
     h5_path = out_dir / "dictionary.h5"[m
     [m
     with h5py.File(h5_path, "w") as h5:[m
[31m-        # Features[m
         feat_grp = h5.create_group("features")[m
         feat_grp.attrs["k"] = cfg.k_features[m
         feat_grp.attrs["definition"] = "r_n = M_n^2 / M_0^2, n=1..k"[m
         feat_grp.create_dataset("feature_names", [m
                                 data=np.array(best_result["feature_names"], dtype="S"))[m
         [m
[31m-        # Model[m
         model_grp = h5.create_group("model")[m
         model_grp.attrs["type"] = best_model[m
         model_grp.attrs["n_params"] = best_result["n_params"][m
[36m@@ -786,7 +1107,6 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
         model_grp.create_dataset("beta_bootstrap_mean", data=bootstrap["beta_mean"])[m
         model_grp.create_dataset("beta_bootstrap_std", data=bootstrap["beta_std"])[m
         [m
[31m-        # Selection[m
         sel_grp = h5.create_group("selection")[m
         sel_grp.attrs["criterion"] = "BIC"[m
         sel_grp.attrs["best_model"] = best_model[m
[36m@@ -797,20 +1117,18 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
             m_grp.attrs["cv_rmse"] = result["cv"]["cv_rmse_mean"][m
             m_grp.attrs["r2"] = result["metrics"]["r2"][m
         [m
[31m-        # Uncertainty[m
         unc_grp = h5.create_group("uncertainty")[m
         unc_grp.attrs["method"] = "bootstrap"[m
         unc_grp.attrs["n_bootstrap"] = bootstrap["n_bootstrap"][m
         unc_grp.attrs["seed"] = cfg.random_seed[m
         unc_grp.create_dataset("sigma_delta", data=bootstrap["sigma_delta"])[m
         [m
[31m-        # Predictions (para reproducibilidad)[m
         pred_grp = h5.create_group("predictions")[m
         pred_grp.create_dataset("delta_true", data=spectrum["delta_uv"])[m
         pred_grp.create_dataset("delta_pred", [m
                                 data=best_result["cv"]["cv_predictions"])[m
         [m
[31m-        # Metadata[m
[32m+[m[32m        h5.attrs["version"] = __version__[m
         h5.attrs["created"] = datetime.now(timezone.utc).isoformat()[m
         h5.attrs["spectrum_source"] = cfg.spectrum_file[m
     [m
[36m@@ -831,13 +1149,13 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
             "tau_delta": cfg.tau_delta,[m
             "sigma_factor": cfg.sigma_factor,[m
             "sigma_cap": cfg.sigma_cap,[m
[31m-            "note": "STRONG_PASS requiere error ≤ τ. WEAK_PASS requiere error ≤ 2σ con σ ≤ σ_cap.",[m
[32m+[m[32m            "note": "STRONG_PASS: error <= tau. WEAK_PASS: error <= 2*sigma con sigma <= sigma_cap.",[m
         },[m
         "sigma_comparison": contracts["c1_sigma"],[m
         "epsilon_comparison": contracts["c1_epsilon"],[m
[31m-        "interpretation": "Hard-wall baseline: Δ es parámetro de barrido, no predicción. "[m
[31m-                          "C1 aquí valida infraestructura, no física. "[m
[31m-                          "Comparación real requiere geometrías emergentes (Bloque A no trivial).",[m
[32m+[m[32m        "interpretation": "Hard-wall baseline: Delta es parametro de barrido, no prediccion. "[m
[32m+[m[32m                          "C1 valida infraestructura, no fisica. "[m
[32m+[m[32m                          "Comparacion real requiere geometrias emergentes (Bloque A no trivial).",[m
     }[m
     with open(ising_path, "w") as f:[m
         json.dump(ising_data, f, indent=2)[m
[36m@@ -845,26 +1163,19 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
     # --- validation.json ---[m
     val_path = out_dir / "validation.json"[m
     [m
[31m-    # Determinar status de C3[m
[31m-    c3_ok = contracts["c3"]["spectral_ok"] if contracts["c3"]["spectral_ok"] is not None else None[m
[31m-    c3_status = contracts["c3"]["status"][m
[31m-    [m
[31m-    # Lógica de hard contracts:[m
[31m-    # - C2 siempre es hard[m
[31m-    # - C3 es hard solo si está activo (no SKIP)[m
[31m-    # - C1 es hard solo si IN_DOMAIN (OUT_OF_DOMAIN no cuenta)[m
     c1_sigma_in_domain = contracts["c1_sigma"].get("in_domain", True)[m
     c1_epsilon_in_domain = contracts["c1_epsilon"].get("in_domain", True)[m
     c1_sigma_status = contracts["c1_sigma"]["global_status"][m
     c1_epsilon_status = contracts["c1_epsilon"]["global_status"][m
     [m
[31m-    # C1 hard fail solo si está in_domain y FAIL[m
[32m+[m[32m    c3 = contracts["c3"][m
[32m+[m[32m    c3_status = c3.get("status", "SKIP")[m
[32m+[m[32m    c3_ok = c3.get("spectral_ok", None)[m
[32m+[m[41m    [m
     c1_hard_fail = ([m
         (c1_sigma_in_domain and c1_sigma_status == "FAIL") or[m
         (c1_epsilon_in_domain and c1_epsilon_status == "FAIL")[m
     )[m
[31m-    [m
[31m-    # C3 hard fail solo si está activo y FAIL[m
     c3_hard_fail = (c3_status != "SKIP" and c3_ok is False)[m
     [m
     all_hard_pass = ([m
[36m@@ -873,11 +1184,11 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
         not c3_hard_fail[m
     )[m
     [m
[31m-    # Rango de Δ del dataset[m
     delta_min = float(spectrum["delta_uv"].min())[m
     delta_max = float(spectrum["delta_uv"].max())[m
     [m
     validation = {[m
[32m+[m[32m        "version": __version__,[m
         "metadata": {[m
             "run": cfg.run,[m
             "spectrum_source": cfg.spectrum_file,[m
[36m@@ -889,7 +1200,7 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
         "C1_ising": {[m
             "description": "Compatibilidad puntual con Ising 3D",[m
             "sigma_cap": cfg.sigma_cap,[m
[31m-            "note": "OUT_OF_DOMAIN = target fuera del rango; STRONG_PASS = dentro de τ; WEAK_PASS = dentro de 2σ con σ≤σ_cap; FAIL = ninguno",[m
[32m+[m[32m            "note": "OUT_OF_DOMAIN = target fuera del rango; STRONG_PASS = dentro de tau; WEAK_PASS = dentro de 2sigma con sigma<=sigma_cap; FAIL = ninguno",[m
             "delta_range": [delta_min, delta_max],[m
             "sigma": contracts["c1_sigma"],[m
             "epsilon": contracts["c1_epsilon"],[m
[36m@@ -898,10 +1209,7 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
             "description": "Consistencia interna del diccionario",[m
             **contracts["c2"],[m
         },[m
[31m-        "C3_spectral": {[m
[31m-            "description": "Compatibilidad espectral (test de ciclo r→Δ→r)",[m
[31m-            **contracts["c3"],[m
[31m-        },[m
[32m+[m[32m        "C3_spectral": c3,[m
         "overall": {[m
             "C1_sigma_status": c1_sigma_status,[m
             "C1_sigma_in_domain": c1_sigma_in_domain,[m
[36m@@ -909,6 +1217,7 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
             "C1_epsilon_in_domain": c1_epsilon_in_domain,[m
             "C2_ok": contracts["c2"]["consistency_ok"],[m
             "C3_status": c3_status,[m
[32m+[m[32m            "C3_failure_mode": c3.get("failure_mode"),[m
             "hard_contracts_definition": "C2 siempre; C3 si activo; C1 solo si IN_DOMAIN",[m
             "all_hard_contracts_pass": all_hard_pass,[m
         },[m
[36m@@ -917,31 +1226,11 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
         json.dump(validation, f, indent=2)[m
     [m
     # --- stage_summary.json ---[m
[31m-    [m
[31m-    # Re-calcular lógica de contratos para summary[m
[31m-    c1_sigma_in_domain = contracts["c1_sigma"].get("in_domain", True)[m
[31m-    c1_epsilon_in_domain = contracts["c1_epsilon"].get("in_domain", True)[m
[31m-    c1_sigma_status_sum = contracts["c1_sigma"]["global_status"][m
[31m-    c1_epsilon_status_sum = contracts["c1_epsilon"]["global_status"][m
[31m-    c3_ok_sum = contracts["c3"]["spectral_ok"] if contracts["c3"]["spectral_ok"] is not None else None[m
[31m-    c3_status_sum = contracts["c3"]["status"][m
[31m-    [m
[31m-    c1_hard_fail_sum = ([m
[31m-        (c1_sigma_in_domain and c1_sigma_status_sum == "FAIL") or[m
[31m-        (c1_epsilon_in_domain and c1_epsilon_status_sum == "FAIL")[m
[31m-    )[m
[31m-    c3_hard_fail_sum = (c3_status_sum != "SKIP" and c3_ok_sum is False)[m
[31m-    [m
[31m-    all_hard_pass_sum = ([m
[31m-        contracts["c2"]["consistency_ok"] and[m
[31m-        not c1_hard_fail_sum and[m
[31m-        not c3_hard_fail_sum[m
[31m-    )[m
[31m-    [m
     summary = {[m
         "stage": "04_diccionario",[m
[31m-        "version": "1.3.0",[m
[32m+[m[32m        "version": __version__,[m
         "created": datetime.now(timezone.utc).isoformat(),[m
[32m+[m[32m        "notes": "El modelo directo Δ→ratios es un proxy suave local usado solo para evaluar invertibilidad (C3). No es física; fallos en C3a indican proxy/modelo insuficiente.",[m
         "config": {[m
             "run": cfg.run,[m
             "spectrum_file": cfg.spectrum_file,[m
[36m@@ -953,13 +1242,16 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
             "tau_delta": cfg.tau_delta,[m
             "sigma_cap": cfg.sigma_cap,[m
             "enable_c3": cfg.enable_c3,[m
[31m-            "direct_model": cfg.direct_model,[m
[31m-            "direct_degree": cfg.direct_degree,[m
             "c3_metric": cfg.c3_metric,[m
[32m+[m[32m            "c3_weights": cfg.c3_weights,[m
             "c3_threshold": cfg.c3_threshold,[m
[32m+[m[32m            "c3_adaptive_threshold": cfg.c3_adaptive_threshold,[m
[32m+[m[32m            "c3_threshold_factor": cfg.c3_threshold_factor,[m
[32m+[m[32m            "direct_model": cfg.direct_model,[m
[32m+[m[32m            "direct_degree": cfg.direct_degree,[m
         },[m
         "data": {[m
[31m-            "delta_range": [float(spectrum["delta_uv"].min()), float(spectrum["delta_uv"].max())],[m
[32m+[m[32m            "delta_range": [delta_min, delta_max],[m
             "n_delta": spectrum["n_delta"],[m
             "n_modes": spectrum["n_modes"],[m
         },[m
[36m@@ -977,20 +1269,20 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
             "sigma_delta_max": bootstrap["sigma_delta_max"],[m
         },[m
         "validation_summary": {[m
[31m-            "C1_sigma_status": c1_sigma_status_sum,[m
[32m+[m[32m            "C1_sigma_status": c1_sigma_status,[m
             "C1_sigma_in_domain": c1_sigma_in_domain,[m
[31m-            "C1_epsilon_status": c1_epsilon_status_sum,[m
[32m+[m[32m            "C1_epsilon_status": c1_epsilon_status,[m
             "C1_epsilon_in_domain": c1_epsilon_in_domain,[m
             "C2_consistency_ok": contracts["c2"]["consistency_ok"],[m
[31m-            "C3_status": c3_status_sum,[m
[31m-            "all_hard_contracts_pass": all_hard_pass_sum,[m
[32m+[m[32m            "C3_status": c3_status,[m
[32m+[m[32m            "C3_failure_mode": c3.get("failure_mode"),[m
[32m+[m[32m            "all_hard_contracts_pass": all_hard_pass,[m
         },[m
[31m-        "hashes": {},  # Se llena después[m
[32m+[m[32m        "hashes": {},[m
     }[m
     [m
     summary_path = out_dir / "stage_summary.json"[m
     [m
[31m-    # Calcular hashes[m
     summary["hashes"] = {[m
         "dictionary.h5": compute_file_hash(h5_path),[m
         "atlas.json": compute_file_hash(atlas_path),[m
[36m@@ -1004,6 +1296,7 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
     # --- manifest.json ---[m
     manifest = {[m
         "stage": "04_diccionario",[m
[32m+[m[32m        "version": __version__,[m
         "run": cfg.run,[m
         "created": datetime.now(timezone.utc).isoformat(),[m
         "files": {[m
[36m@@ -1034,8 +1327,52 @@[m [mdef write_outputs(out_dir: Path, cfg: Config, spectrum: dict,[m
 # Main[m
 # =============================================================================[m
 [m
[32m+[m[32mdef run_self_test() -> int:[m
[32m+[m[32m    """Self-tests deterministas para validar lógica interna de C3.[m
[32m+[m
[32m+[m[32m    T1 (inyectivo): r = Δ -> C3a PASS, C3b PASS[m
[32m+[m[32m    T2 (no-inyectivo): r = (Δ-5)^2 -> C3a PASS, C3b FAIL/WARN[m
[32m+[m[32m    """[m
[32m+[m[32m    print("Running self-test...")[m
[32m+[m[32m    np.random.seed(0)[m
[32m+[m
[32m+[m[32m    # T1: inyectivo[m
[32m+[m[32m    Delta = np.linspace(1.0, 10.0, 80)[m
[32m+[m[32m    X = Delta.reshape(-1, 1)[m
[32m+[m[32m    cfg = Config(run="__selftest__", test_mode=True, k_features=1, enable_c3=True, direct_model="poly", direct_degree=1)[m
[32m+[m[32m    res1 = evaluate_c3_full(X, Delta, cfg, inverse_model_type="linear")[m
[32m+[m[32m    if not (res1.get("c3a_status") == "PASS" and res1.get("c3b_status") == "PASS" and res1.get("status") == "PASS"):[m
[32m+[m[32m        print(f"FATAL: Self-test T1 fallo (inyectivo). {res1}")[m
[32m+[m[32m        return 1[m
[32m+[m
[32m+[m[32m    # T2: no-inyectivo (controlado)[m
[32m+[m[32m    r = (Delta - 5.0) ** 2[m
[32m+[m[32m    X2 = r.reshape(-1, 1)[m
[32m+[m[32m    cfg2 = Config(run="__selftest__", test_mode=True, k_features=1, enable_c3=True, direct_model="poly", direct_degree=2)[m
[32m+[m[32m    res2 = evaluate_c3_full(X2, Delta, cfg2, inverse_model_type="poly2")[m
[32m+[m[32m    # Esperado: proxy aprende (C3a PASS) pero el ciclo se rompe por ambiguedad (C3b FAIL)[m
[32m+[m[32m    if not (res2.get("c3a_status") == "PASS" and res2.get("c3b_status") == "FAIL" and res2.get("status") == "FAIL" and res2.get("failure_mode") == "CYCLE_INCONSISTENT"):[m
[32m+[m[32m        print(f"FATAL: Self-test T2 fallo (no-inyectivo controlado). {res2}")[m
[32m+[m[32m        return 1[m
[32m+[m
[32m+[m[32m    print("Self-test PASSED.")[m
[32m+[m[32m    return 0[m
[32m+[m
[32m+[m
[32m+[m[32m# =============================================================================[m
[32m+[m[32m# Main[m
[32m+[m[32m# =============================================================================[m
[32m+[m
[32m+[m
 def main() -> int:[m
     cfg = parse_args()[m
[32m+[m
[32m+[m[32m    if cfg.test_mode:[m
[32m+[m[32m        return run_self_test()[m
[32m+[m
[32m+[m[32m    if cfg.run is None:[m
[32m+[m[32m        print("ERROR: --run es requerido", file=sys.stderr)[m
[32m+[m[32m        return 1[m
     [m
     # Cargar espectro[m
     spec_path = Path("runs") / cfg.run / "spectrum" / cfg.spectrum_file[m
[36m@@ -1047,9 +1384,10 @@[m [mdef main() -> int:[m
     [m
     spectrum = load_spectrum(spec_path)[m
     [m
[32m+[m[32m    print(f"Bloque C v{__version__}")[m
     print(f"Espectro cargado: {spec_path}")[m
     print(f"  n_delta={spectrum['n_delta']}, n_modes={spectrum['n_modes']}")[m
[31m-    print(f"  Δ ∈ [{spectrum['delta_uv'].min():.3f}, {spectrum['delta_uv'].max():.3f}]")[m
[32m+[m[32m    print(f"  Delta in [{spectrum['delta_uv'].min():.3f}, {spectrum['delta_uv'].max():.3f}]")[m
     print()[m
     [m
     # Validar k_features[m
[36m@@ -1067,13 +1405,13 @@[m [mdef main() -> int:[m
     print()[m
     [m
     # --- Selección de modelo ---[m
[31m-    print("Evaluando modelos...")[m
[32m+[m[32m    print("Evaluando modelos inversos (r -> Delta)...")[m
     model_selection = select_best_model(X, y, cfg.models, cfg.cv_folds, cfg.random_seed)[m
     [m
     for model_type, result in model_selection["all_models"].items():[m
[31m-        marker = "→" if model_type == model_selection["best_model"] else " "[m
[32m+[m[32m        marker = "->" if model_type == model_selection["best_model"] else "  "[m
         print(f"  {marker} {model_type:8s}: BIC={result['info_criteria']['bic']:8.2f}, "[m
[31m-              f"CV RMSE={result['cv']['cv_rmse_mean']:.4f}, R²={result['metrics']['r2']:.4f}")[m
[32m+[m[32m              f"CV RMSE={result['cv']['cv_rmse_mean']:.4f}, R2={result['metrics']['r2']:.4f}")[m
     [m
     best_model = model_selection["best_model"][m
     best_result = model_selection["all_models"][best_model][m
[36m@@ -1081,10 +1419,10 @@[m [mdef main() -> int:[m
     print()[m
     [m
     # --- Bootstrap para incertidumbre ---[m
[31m-    print(f"Bootstrap ({cfg.n_bootstrap} réplicas)...")[m
[32m+[m[32m    print(f"Bootstrap ({cfg.n_bootstrap} replicas)...")[m
     bootstrap = bootstrap_uncertainty(X, y, best_model, cfg.n_bootstrap, cfg.random_seed)[m
[31m-    print(f"  σ_Δ medio: {bootstrap['sigma_delta_mean']:.4f}")[m
[31m-    print(f"  σ_Δ máximo: {bootstrap['sigma_delta_max']:.4f}")[m
[32m+[m[32m    print(f"  sigma_Delta medio: {bootstrap['sigma_delta_mean']:.4f}")[m
[32m+[m[32m    print(f"  sigma_Delta maximo: {bootstrap['sigma_delta_max']:.4f}")[m
     print()[m
     [m
     # --- Predicciones finales ---[m
[36m@@ -1095,10 +1433,9 @@[m [mdef main() -> int:[m
     # --- Contratos ---[m
     print("Evaluando contratos...")[m
     [m
[31m-    # Rango de Δ del dataset (para detectar OUT_OF_DOMAIN)[m
     delta_range = (float(y.min()), float(y.max()))[m
     [m
[31m-    # C1: Ising (con sigma_cap y delta_range)[m
[32m+[m[32m    # C1: Ising[m
     c1_sigma = evaluate_c1_ising([m
         y_pred, bootstrap["sigma_delta"],[m
         cfg.delta_sigma, cfg.tau_delta, cfg.sigma_factor, "sigma",[m
[36m@@ -1113,19 +1450,11 @@[m [mdef main() -> int:[m
     # C2: Consistencia[m
     c2 = evaluate_c2_consistency(y, y_pred, best_result["cv"]["cv_rmse_mean"])[m
     [m
[31m-    # C3: Espectral (completo si --enable-c3, placeholder si no)[m
[32m+[m[32m    # C3: Espectral (con diagnóstico causal)[m
     if cfg.enable_c3:[m
[31m-        print(f"  C3: Evaluando modelo directo (model={cfg.direct_model}, degree={cfg.direct_degree}, metric={cfg.c3_metric})...")[m
[31m-        c3 = evaluate_c3_cycle([m
[31m-            X, y, best_model,[m
[31m-            direct_model_type=cfg.direct_model,[m
[31m-            direct_degree=cfg.direct_degree,[m
[31m-            c3_metric=cfg.c3_metric,[m
[31m-            c3_threshold=cfg.c3_threshold[m
[31m-        )[m
[31m-    else:[m
[31m-        c3 = evaluate_c3_spectral(X, X, metric=cfg.c3_metric, threshold=cfg.c3_threshold, is_placeholder=True)[m
[31m-[m
[32m+[m[32m        print(f"  C3: Modelo directo (degree={cfg.direct_degree}), metric={cfg.c3_metric}, weights={cfg.c3_weights}")[m
[32m+[m[32m    c3 = evaluate_c3_full(X, y, cfg, best_model)[m
[32m+[m[41m    [m
     contracts = {[m
         "c1_sigma": c1_sigma,[m
         "c1_epsilon": c1_epsilon,[m
[36m@@ -1133,28 +1462,45 @@[m [mdef main() -> int:[m
         "c3": c3,[m
     }[m
     [m
[31m-    # Prints mejorados con status[m
[32m+[m[32m    # Prints[m
     sigma_status = c1_sigma['global_status'][m
     epsilon_status = c1_epsilon['global_status'][m
     [m
     if sigma_status == "OUT_OF_DOMAIN":[m
[31m-        print(f"  C1 (Ising σ): OUT_OF_DOMAIN (Δ_σ={cfg.delta_sigma:.4f} < Δ_min={delta_range[0]:.3f})")[m
[32m+[m[32m        print(f"  C1 (Ising sigma): OUT_OF_DOMAIN (Delta_sigma={cfg.delta_sigma:.4f} < Delta_min={delta_range[0]:.3f})")[m
     else:[m
[31m-        print(f"  C1 (Ising σ): {sigma_status:12s} "[m
[32m+[m[32m        print(f"  C1 (Ising sigma): {sigma_status:12s} "[m
               f"(error: {c1_sigma['best_candidate']['error']:.4f}, "[m
[31m-              f"σ: {c1_sigma['best_candidate']['sigma_delta']:.4f})")[m
[32m+[m[32m              f"sigma: {c1_sigma['best_candidate']['sigma_delta']:.4f})")[m
     [m
     if epsilon_status == "OUT_OF_DOMAIN":[m
[31m-        print(f"  C1 (Ising ε): OUT_OF_DOMAIN (Δ_ε={cfg.delta_epsilon:.4f} < Δ_min={delta_range[0]:.3f})")[m
[32m+[m[32m        print(f"  C1 (Ising epsilon): OUT_OF_DOMAIN (Delta_epsilon={cfg.delta_epsilon:.4f} < Delta_min={delta_range[0]:.3f})")[m
     else:[m
[31m-        print(f"  C1 (Ising ε): {epsilon_status:12s} "[m
[32m+[m[32m        print(f"  C1 (Ising epsilon): {epsilon_status:12s} "[m
               f"(error: {c1_epsilon['best_candidate']['error']:.4f}, "[m
[31m-              f"σ: {c1_epsilon['best_candidate']['sigma_delta']:.4f})")[m
[32m+[m[32m              f"sigma: {c1_epsilon['best_candidate']['sigma_delta']:.4f})")[m
     [m
     print(f"  C2 (consistencia): {'PASS' if c2['consistency_ok'] else 'FAIL':12s} "[m
           f"(CV RMSE: {c2['cv_rmse']:.4f})")[m
[31m-    print(f"  C3 (espectral): {c3['status']:12s}"[m
[31m-          + (f" (RMSE: {c3['rmse_global']:.4f})" if c3['rmse_global'] is not None else ""))[m
[32m+[m[41m    [m
[32m+[m[32m    c3_status = c3.get("status", "SKIP")[m
[32m+[m[32m    if c3_status == "SKIP":[m
[32m+[m[32m        print(f"  C3 (espectral): SKIP (usar --enable-c3)")[m
[32m+[m[32m    else:[m
[32m+[m[32m        failure_mode = c3.get("failure_mode", "")[m
[32m+[m[32m        c3a_val = c3.get("c3a_decoder", {}).get("global", 0)[m
[32m+[m[32m        c3b_val = c3.get("c3b_cycle", {}).get("global", 0)[m
[32m+[m[32m        threshold_eff = c3.get("threshold", {}).get("effective", cfg.c3_threshold)[m
[32m+[m[32m        noise_floor = c3.get("noise_floor", {}).get("median_sigma", 0)[m
[32m+[m[41m        [m
[32m+[m[32m        print(f"  C3 (espectral): {c3_status:12s} "[m
[32m+[m[32m              f"[C3a={c3a_val:.4f}, C3b={c3b_val:.4f}, threshold={threshold_eff:.4f}]")[m
[32m+[m[32m        if failure_mode:[m
[32m+[m[32m            print(f"      failure_mode: {failure_mode}")[m
[32m+[m[32m        print(f"      noise_floor: {noise_floor:.6f}, metric: {cfg.c3_metric}")[m
[32m+[m[32m        if cfg.c3_weights != "none":[m
[32m+[m[32m            print(f"      weights: {cfg.c3_weights}")[m
[32m+[m[41m    [m
     print()[m
     [m
     # --- Atlas ---[m
[36m@@ -1170,23 +1516,26 @@[m [mdef main() -> int:[m
     print()[m
     [m
     # --- Resumen ---[m
[31m-    c3_status = contracts["c3"]["status"][m
[31m-    c3_ok = contracts["c3"]["spectral_ok"][m
[31m-    [m
[31m-    # Resultado final basado en contratos duros (C2, C3 si está activo)[m
[31m-    hard_pass = c2["consistency_ok"] and (c3_ok if c3_ok is not None else True)[m
[32m+[m[32m    c3_ok = c3.get("spectral_ok", True) if c3_status != "SKIP" else True[m
[32m+[m[32m    hard_pass = c2["consistency_ok"] and c3_ok[m
     [m
     if hard_pass:[m
[31m-        print("✓ VERIFICADO: Diccionario consistente")[m
[32m+[m[32m        print("[OK] VERIFICADO: Diccionario consistente")[m
         if c3_status == "SKIP":[m
             print("  (C3 no evaluado - usar --enable-c3 para test de ciclo completo)")[m
         return 0[m
     else:[m
[31m-        print("✗ FALLO en contratos duros:")[m
[32m+[m[32m        print("[FAIL] FALLO en contratos duros:")[m
         if not c2["consistency_ok"]:[m
             print(f"  - C2: CV RMSE = {c2['cv_rmse']:.4f} > 0.05")[m
[31m-        if c3_ok is not None and not c3_ok:[m
[31m-            print(f"  - C3: RMSE ciclo = {contracts['c3']['rmse_global']:.4f}")[m
[32m+[m[32m        if not c3_ok:[m
[32m+[m[32m            failure = c3.get("failure_mode", "unknown")[m
[32m+[m[32m            print(f"  - C3: {failure}")[m
[32m+[m[32m            if "c3a_decoder" in c3:[m
[32m+[m[32m                print(f"    C3a (decoder): {c3['c3a_decoder']['global']:.4f}")[m
[32m+[m[32m            if "c3b_cycle" in c3:[m
[32m+[m[32m                print(f"    C3b (cycle):   {c3['c3b_cycle']['global']:.4f}")[m
[32m+[m[32m            print(f"    threshold:     {c3['threshold']['effective']:.4f}")[m
         return 1[m
 [m
 [m
