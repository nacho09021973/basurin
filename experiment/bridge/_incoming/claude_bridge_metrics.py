#!/usr/bin/env python3
"""
bridge_metrics.py — Métricas para Bridge Discovery (F4-1)

Funciones para evaluar calidad estructural de puentes entre espacios de features.

Métricas:
- kNN_preservation: preservación de vecindarios bajo el puente
- degeneracy_index: fracción de puntos que colapsan
- bootstrap_stability: CV de métricas bajo remuestreo
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
from scipy.spatial.distance import cdist


def normalize_features(X: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normaliza features a media 0 y std 1.
    
    Returns:
        X_norm: array normalizado
        mean: medias usadas
        std: desviaciones estándar usadas
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < eps, eps, std)  # Evitar división por cero
    X_norm = (X - mean) / std
    return X_norm, mean, std


def get_knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Calcula índices de k vecinos más cercanos para cada punto.
    
    Args:
        X: array (n_samples, n_features)
        k: número de vecinos
    
    Returns:
        indices: array (n_samples, k) con índices de vecinos
    """
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)  # No más vecinos que puntos disponibles
    
    # Matriz de distancias
    D = cdist(X, X, metric='euclidean')
    
    # Para cada punto, encontrar k vecinos más cercanos (excluyendo a sí mismo)
    indices = np.zeros((n_samples, k), dtype=np.int64)
    for i in range(n_samples):
        # Ordenar por distancia, excluir el punto mismo (distancia 0)
        sorted_idx = np.argsort(D[i])
        # Tomar los k siguientes (el primero es él mismo)
        indices[i] = sorted_idx[1:k+1]
    
    return indices


def knn_preservation(X_source: np.ndarray, X_target: np.ndarray, k: int = 5) -> dict:
    """Calcula preservación de vecindarios kNN entre dos espacios.
    
    Args:
        X_source: puntos en espacio fuente (n, d1)
        X_target: puntos en espacio objetivo (n, d2)
        k: número de vecinos
    
    Returns:
        dict con overlap_mean, overlap_per_point, k_used
    """
    n_samples = X_source.shape[0]
    if n_samples < 3:
        return {
            "overlap_mean": 0.0,
            "overlap_per_point": [],
            "k_used": 0,
            "error": "too_few_samples"
        }
    
    k = min(k, n_samples - 1)
    
    # Vecinos en cada espacio
    knn_source = get_knn_indices(X_source, k)
    knn_target = get_knn_indices(X_target, k)
    
    # Calcular overlap para cada punto
    overlaps = np.zeros(n_samples)
    for i in range(n_samples):
        set_source = set(knn_source[i])
        set_target = set(knn_target[i])
        overlaps[i] = len(set_source & set_target) / k
    
    return {
        "overlap_mean": float(np.mean(overlaps)),
        "overlap_std": float(np.std(overlaps)),
        "overlap_min": float(np.min(overlaps)),
        "overlap_max": float(np.max(overlaps)),
        "overlap_per_point": overlaps.tolist(),
        "k_used": k,
    }


def degeneracy_index(X: np.ndarray, eps_factor: float = 0.1) -> dict:
    """Calcula índice de degeneración (puntos colapsados).
    
    Args:
        X: proyección en espacio común (n, d)
        eps_factor: fracción del diámetro para definir "colapso"
    
    Returns:
        dict con degeneracy_index, n_degenerate, eps_used
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return {
            "degeneracy_index": 0.0,
            "n_degenerate": 0,
            "eps_used": 0.0,
            "error": "too_few_samples"
        }
    
    # Matriz de distancias
    D = cdist(X, X, metric='euclidean')
    
    # Diámetro del espacio (máxima distancia)
    diameter = np.max(D)
    if diameter < 1e-10:
        return {
            "degeneracy_index": 1.0,
            "n_degenerate": n_samples,
            "eps_used": 0.0,
            "note": "all_points_collapsed"
        }
    
    eps = eps_factor * diameter
    
    # Contar puntos degenerados (tienen otro punto a distancia < eps)
    np.fill_diagonal(D, np.inf)  # Ignorar distancia a sí mismo
    min_distances = np.min(D, axis=1)
    degenerate_mask = min_distances < eps
    n_degenerate = np.sum(degenerate_mask)
    
    return {
        "degeneracy_index": float(n_degenerate / n_samples),
        "n_degenerate": int(n_degenerate),
        "n_total": int(n_samples),
        "eps_used": float(eps),
        "diameter": float(diameter),
        "min_distance_median": float(np.median(min_distances)),
        "degenerate_indices": np.where(degenerate_mask)[0].tolist(),
    }


def cca_bridge(X_A: np.ndarray, X_B: np.ndarray, 
               n_components: Optional[int] = None) -> dict:
    """Aplica CCA para encontrar proyecciones correlacionadas.
    
    Args:
        X_A: puntos espacio A (n, d_A)
        X_B: puntos espacio B (n, d_B)
        n_components: dimensiones del espacio común (default: min(d_A, d_B))
    
    Returns:
        dict con A_proj, B_proj, canonical_correlations, loadings
    """
    from sklearn.cross_decomposition import CCA
    
    n_samples, d_A = X_A.shape
    _, d_B = X_B.shape
    
    if n_components is None:
        n_components = min(d_A, d_B, n_samples - 1)
    n_components = min(n_components, d_A, d_B, n_samples - 1)
    
    # Ajustar CCA
    cca = CCA(n_components=n_components, max_iter=1000)
    try:
        A_proj, B_proj = cca.fit_transform(X_A, X_B)
    except Exception as e:
        return {
            "error": str(e),
            "A_proj": None,
            "B_proj": None,
        }
    
    # Calcular correlaciones canónicas
    canonical_corrs = []
    for i in range(n_components):
        corr = np.corrcoef(A_proj[:, i], B_proj[:, i])[0, 1]
        canonical_corrs.append(float(corr) if np.isfinite(corr) else 0.0)
    
    # Varianza explicada (aproximada)
    var_A_total = np.var(X_A, axis=0).sum()
    var_A_proj = np.var(A_proj, axis=0).sum()
    var_B_total = np.var(X_B, axis=0).sum()
    var_B_proj = np.var(B_proj, axis=0).sum()
    
    return {
        "A_proj": A_proj,
        "B_proj": B_proj,
        "canonical_correlations": canonical_corrs,
        "n_components": n_components,
        "explained_variance_A": float(var_A_proj / max(var_A_total, 1e-10)),
        "explained_variance_B": float(var_B_proj / max(var_B_total, 1e-10)),
        "loadings_A": cca.x_weights_.tolist() if hasattr(cca, 'x_weights_') else None,
        "loadings_B": cca.y_weights_.tolist() if hasattr(cca, 'y_weights_') else None,
    }


def bootstrap_stability(X_A: np.ndarray, X_B: np.ndarray, 
                        metric_fn: callable, 
                        n_bootstrap: int = 100,
                        seed: int = 42) -> dict:
    """Calcula estabilidad de una métrica vía bootstrap.
    
    Args:
        X_A, X_B: datos emparejados
        metric_fn: función que toma (X_A, X_B) y retorna float
        n_bootstrap: número de muestras bootstrap
        seed: semilla para reproducibilidad
    
    Returns:
        dict con mean, std, cv, samples
    """
    rng = np.random.RandomState(seed)
    n_samples = X_A.shape[0]
    
    values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        val = metric_fn(X_A[idx], X_B[idx])
        if np.isfinite(val):
            values.append(val)
    
    if len(values) < 5:
        return {
            "mean": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "n_valid": len(values),
            "error": "insufficient_valid_samples"
        }
    
    values = np.array(values)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    cv = std_val / max(abs(mean_val), 1e-10)
    
    return {
        "mean": mean_val,
        "std": std_val,
        "cv": float(cv),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "n_bootstrap": n_bootstrap,
        "n_valid": len(values),
        "samples": values.tolist(),
    }


def generate_negative_control(X: np.ndarray, mode: str = "permute", 
                               seed: int = 42) -> np.ndarray:
    """Genera control negativo.
    
    Args:
        X: datos originales
        mode: "permute" (permutación de filas) o "noise" (ruido gaussiano)
        seed: semilla
    
    Returns:
        X_control: datos de control
    """
    rng = np.random.RandomState(seed)
    
    if mode == "permute":
        # Permutación aleatoria de filas (rompe correspondencia)
        idx = rng.permutation(X.shape[0])
        return X[idx].copy()
    elif mode == "noise":
        # Ruido gaussiano con misma estadística marginal
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-10
        return rng.normal(mean, std, size=X.shape)
    else:
        raise ValueError(f"mode desconocido: {mode}")


def evaluate_bridge_full(X_A: np.ndarray, X_B: np.ndarray,
                          k_neighbors: int = 5,
                          n_bootstrap: int = 100,
                          seed: int = 42) -> dict:
    """Evaluación completa del puente con todos los contratos C7.
    
    Args:
        X_A: features espacio A (atlas)
        X_B: features espacio B (externo)
        k_neighbors: k para kNN preservation
        n_bootstrap: muestras bootstrap
        seed: semilla
    
    Returns:
        dict con todos los contratos y diagnósticos
    """
    # Normalizar
    A_norm, _, _ = normalize_features(X_A)
    B_norm, _, _ = normalize_features(X_B)
    
    # CCA bridge
    bridge = cca_bridge(A_norm, B_norm)
    if bridge.get("error"):
        return {
            "status": "ERROR",
            "error": bridge["error"],
            "contracts": {}
        }
    
    A_proj = bridge["A_proj"]
    B_proj = bridge["B_proj"]
    
    # C7a: Estructura (kNN preservation)
    knn_result = knn_preservation(A_proj, B_proj, k=k_neighbors)
    knn_value = knn_result["overlap_mean"]
    
    # C7b: Degeneración
    degen_A = degeneracy_index(A_proj)
    degen_B = degeneracy_index(B_proj)
    degen_value = max(degen_A["degeneracy_index"], degen_B["degeneracy_index"])
    
    # C7c: Estabilidad bootstrap
    def metric_fn(XA, XB):
        XA_n, _, _ = normalize_features(XA)
        XB_n, _, _ = normalize_features(XB)
        b = cca_bridge(XA_n, XB_n, n_components=bridge["n_components"])
        if b.get("error"):
            return np.nan
        knn = knn_preservation(b["A_proj"], b["B_proj"], k=k_neighbors)
        return knn["overlap_mean"]
    
    stability = bootstrap_stability(X_A, X_B, metric_fn, n_bootstrap, seed)
    cv_value = stability["cv"]
    
    # C7d: Control negativo (permutación)
    B_permuted = generate_negative_control(X_B, mode="permute", seed=seed)
    B_perm_norm, _, _ = normalize_features(B_permuted)
    bridge_neg = cca_bridge(A_norm, B_perm_norm, n_components=bridge["n_components"])
    if not bridge_neg.get("error"):
        knn_neg = knn_preservation(bridge_neg["A_proj"], bridge_neg["B_proj"], k=k_neighbors)
        knn_neg_value = knn_neg["overlap_mean"]
    else:
        knn_neg_value = 0.0
    
    # C7e: Control positivo (split del atlas)
    n_half = X_A.shape[0] // 2
    A1, A2 = X_A[:n_half], X_A[n_half:2*n_half]
    if len(A1) >= 5 and len(A2) >= 5:
        A1_n, _, _ = normalize_features(A1)
        A2_n, _, _ = normalize_features(A2)
        bridge_pos = cca_bridge(A1_n, A2_n, n_components=min(A1.shape[1], bridge["n_components"]))
        if not bridge_pos.get("error"):
            knn_pos = knn_preservation(bridge_pos["A_proj"], bridge_pos["B_proj"], k=k_neighbors)
            knn_pos_value = knn_pos["overlap_mean"]
        else:
            knn_pos_value = 0.0
    else:
        knn_pos_value = None
    
    # Umbrales
    THRESHOLD_KNN = 0.3
    THRESHOLD_DEGEN = 0.5
    THRESHOLD_CV = 0.3
    
    # Evaluar contratos
    c7a_pass = knn_value > THRESHOLD_KNN and knn_value > 2 * knn_neg_value
    c7b_pass = degen_value < THRESHOLD_DEGEN
    c7c_pass = cv_value < THRESHOLD_CV if np.isfinite(cv_value) else False
    c7d_pass = knn_neg_value < THRESHOLD_KNN
    c7e_pass = knn_pos_value is None or knn_pos_value > THRESHOLD_KNN
    
    # Determinar status global y failure_mode
    if not c7d_pass:
        status = "FAIL_LEAKAGE"
        failure_mode = "FALSE_POSITIVE"
    elif not c7a_pass and c7d_pass:
        status = "FAIL_STRUCTURE"
        failure_mode = "NO_BRIDGE"
    elif not c7b_pass:
        status = "FAIL_DEGENERACY"
        failure_mode = "DEGENERATE_BRIDGE"
    elif not c7c_pass:
        status = "FAIL_UNSTABLE"
        failure_mode = "UNSTABLE_BRIDGE"
    elif c7a_pass and c7b_pass and c7c_pass and c7d_pass:
        status = "PASS"
        failure_mode = None
    else:
        status = "FAIL"
        failure_mode = "UNKNOWN"
    
    return {
        "status": status,
        "failure_mode": failure_mode,
        "contracts": {
            "C7a_structure": {
                "status": "PASS" if c7a_pass else "FAIL",
                "knn_preservation": knn_value,
                "threshold": THRESHOLD_KNN,
                "vs_control_neg": knn_value / max(knn_neg_value, 1e-10),
            },
            "C7b_degeneracy": {
                "status": "PASS" if c7b_pass else "FAIL",
                "degeneracy_index": degen_value,
                "threshold": THRESHOLD_DEGEN,
                "detail_A": degen_A,
                "detail_B": degen_B,
            },
            "C7c_stability": {
                "status": "PASS" if c7c_pass else "FAIL",
                "bootstrap_cv": cv_value,
                "threshold": THRESHOLD_CV,
                "bootstrap_mean": stability["mean"],
                "bootstrap_std": stability["std"],
            },
            "C7d_no_false_positive": {
                "status": "PASS" if c7d_pass else "FAIL",
                "control_neg_knn": knn_neg_value,
                "threshold": THRESHOLD_KNN,
            },
            "C7e_positive_control": {
                "status": "PASS" if c7e_pass else "WARN",
                "control_pos_knn": knn_pos_value,
                "threshold": THRESHOLD_KNN,
            },
        },
        "diagnostics": {
            "canonical_correlations": bridge["canonical_correlations"],
            "explained_variance_A": bridge["explained_variance_A"],
            "explained_variance_B": bridge["explained_variance_B"],
            "n_components": bridge["n_components"],
            "knn_detail": knn_result,
        },
        "projections": {
            "A_proj": A_proj,
            "B_proj": B_proj,
        },
    }


if __name__ == "__main__":
    # Self-test
    print("Running self-test...")
    np.random.seed(42)
    
    # Test 1: Datos con estructura compartida (no perfectamente correlacionados)
    # Simula dos vistas de un mismo fenómeno con ruido
    n = 60
    latent = np.random.randn(n, 2)  # Espacio latente común
    X = latent @ np.random.randn(2, 4) + 0.3 * np.random.randn(n, 4)  # Vista A
    Y = latent @ np.random.randn(2, 3) + 0.3 * np.random.randn(n, 3)  # Vista B
    
    result = evaluate_bridge_full(X, Y, k_neighbors=5, n_bootstrap=30, seed=42)
    print(f"Test 1 (estructura compartida): {result['status']}")
    print(f"  kNN preservation: {result['contracts']['C7a_structure']['knn_preservation']:.3f}")
    print(f"  Degeneracy: {result['contracts']['C7b_degeneracy']['degeneracy_index']:.3f}")
    # Aceptamos cualquier resultado que no sea FAIL_LEAKAGE
    assert result["status"] != "FAIL_LEAKAGE", f"Leakage detectado (bug en metodología)"
    
    # Test 2: Datos independientes → control negativo debe fallar estructura
    X2 = np.random.randn(n, 4)
    Y2 = np.random.randn(n, 3)  # Independiente
    
    result2 = evaluate_bridge_full(X2, Y2, k_neighbors=5, n_bootstrap=30, seed=42)
    print(f"Test 2 (independiente): {result2['status']}")
    print(f"  kNN preservation: {result2['contracts']['C7a_structure']['knn_preservation']:.3f}")
    # Esperamos que no de PASS (no debe encontrar estructura donde no la hay)
    # Pero tampoco FAIL_LEAKAGE
    assert result2["status"] != "FAIL_LEAKAGE", f"Leakage detectado en datos independientes"
    
    # Test 3: Verificar que control negativo funciona
    print(f"Test 3 (control negativo funciona):")
    print(f"  Control neg kNN (test1): {result['contracts']['C7d_no_false_positive']['control_neg_knn']:.3f}")
    print(f"  Control neg kNN (test2): {result2['contracts']['C7d_no_false_positive']['control_neg_knn']:.3f}")
    
    print()
    print("Self-test COMPLETADO (verificar outputs manualmente).")
