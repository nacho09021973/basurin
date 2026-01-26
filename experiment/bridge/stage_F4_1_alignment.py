#!/usr/bin/env python3
"""BASURIN — Fase 4 / Experimento F4-1

Stage: bridge_f4_1_alignment

Objetivo
--------
Descubrir un "puente" estadístico/topológico entre:
  X: features internas del atlas (p.ej. ratios)
  Y: features externas (p.ej. ringdown/QNM)
SIN imponer un mapa físico a priori.

Implementación
--------------
- Alineación lineal vía CCA (Canonincal Correlation Analysis) sobre datos estandarizados.
- Test de permutación (shuffle de pares) para evaluar significancia (anti-falsos-positivos).
- Estabilidad via bootstrap: coherencia angular de los ejes canónicos.
- Diagnóstico de no-inyectividad: condición local (cond(J_local)) y Var(X|Y-neighborhood).
- Kill-switch anti-leakage: aborta si detecta identidad/colinealidad trivial entre columnas X/Y.

IO (determinista)
-----------------
Escribe exclusivamente bajo:
  runs/<run_id>/bridge_f4_1_alignment/
    manifest.json
    stage_summary.json
    outputs/

Entradas esperadas
------------------
Atlas (JSON):
  - O bien {"points": [{"id": ..., "x": [..]} , ...]}
  - O bien lista [{"id": ..., "x": [..]}, ...]
  - O bien {"X": [[..],[..],...], "ids": [...]} 

Ringdown features (JSON):
  - O bien {"events": [{"id": ..., "y": [..]} , ...]}
  - O bien lista [{"id": ..., "y": [..]}, ...]
  - O bien {"Y": [[..],[..],...], "ids": [...]} 

Pareado:
  - Por defecto une por "id" (intersección).
  - Si no hay ids, requiere --pairing-policy order (y N_X == N_Y).

Nota: Este stage NO dicta PASS/FAIL final (eso es Contrato C7 separado).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# Allow running as a script by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    assert_within_runs,
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)
from experiment.bridge.pairing import pair_frames
__version__ = "0.1.0"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Parsing de inputs
# =============================================================================

def _contract_error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _coerce_vector(vec: Any) -> Optional[List[float]]:
    if isinstance(vec, list):
        return [float(x) for x in vec]
    if isinstance(vec, dict):
        for key in ["values", "vector", "data", "features"]:
            if key in vec:
                return _coerce_vector(vec[key])
    return None


def _load_sklearn() -> None:
    global CCA, NearestNeighbors, StandardScaler
    from sklearn.cross_decomposition import CCA as _CCA
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    CCA = _CCA
    NearestNeighbors = _NearestNeighbors
    StandardScaler = _StandardScaler


def _resolve_feature_key(obj: Any, kind: str, feature_key_hint: Optional[str]) -> str:
    if isinstance(obj, dict):
        meta = obj.get("meta")
        if isinstance(meta, dict) and meta.get("feature_key"):
            return str(meta.get("feature_key"))
    if feature_key_hint:
        return str(feature_key_hint)
    if kind == "atlas":
        return "ratios"
    if kind in {"features", "ringdown"}:
        return "tangentes_locales_v1"
    _contract_error(f"kind inválido: {kind}")
    return ""


def _extract_meta(obj: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return meta
    if "meta" in obj and isinstance(obj["meta"], dict):
        meta.update(obj["meta"])
    for mk in ["feature_key", "k", "dim", "source", "schema_version", "created"]:
        if mk in obj:
            meta[mk] = obj[mk]
    if "columns" in obj:
        meta["columns"] = obj["columns"]
    if "feature_names" in obj and "columns" not in meta:
        meta["columns"] = obj["feature_names"]
    return meta


def _coerce_matrix(rows: Any, path: Path, key_label: str) -> np.ndarray:
    try:
        mat = np.asarray(rows, dtype=float)
    except Exception as exc:
        _contract_error(f"Formato inválido en {path}: '{key_label}' no es matriz numérica ({exc}).")
    if mat.ndim != 2:
        _contract_error(f"Formato inválido en {path}: '{key_label}' debe ser 2D.")
    return mat


def load_feature_json(
    path: Path,
    kind: str,
    feature_key_hint: Optional[str] = None,
) -> Tuple[Optional[List[Any]], np.ndarray, Dict[str, Any]]:
    """Carga JSON y devuelve ids (si existen), matriz y metadatos mínimos."""
    with open(path, "r") as f:
        obj = json.load(f)

    key_vec = "x" if kind == "atlas" else "y"
    feature_key = _resolve_feature_key(obj, kind, feature_key_hint)
    meta = _extract_meta(obj)
    if feature_key and "feature_key" not in meta:
        meta["feature_key"] = feature_key

    if isinstance(obj, dict) and "ids" in obj:
        for candidate in (key_vec, key_vec.upper()):
            if candidate in obj:
                ids = obj["ids"]
                mat = _coerce_matrix(obj[candidate], path, candidate)
                if ids is not None and len(ids) != mat.shape[0]:
                    _contract_error(f"{path}: ids no coincide con filas de '{candidate}'.")
                if "columns" not in meta and meta.get("feature_key"):
                    meta["columns"] = [f"{meta['feature_key']}_{i}" for i in range(mat.shape[1])]
                return ids, mat, meta

    rows = None
    if isinstance(obj, dict):
        for k in ["points", "theories", "events", "rows", "data"]:
            if isinstance(obj.get(k), list):
                rows = obj[k]
                break
    elif isinstance(obj, list):
        rows = obj

    if isinstance(rows, list):
        ids: List[Any] = []
        vectors: List[List[float]] = []
        for row in rows:
            if not isinstance(row, dict):
                _contract_error(f"{path}: filas deben ser dicts con id y '{key_vec}'.")
            rid = row.get("id", row.get("uid", row.get("name")))
            vec = row.get(key_vec, row.get(key_vec.upper()))
            if vec is None and key_vec == "x":
                for fallback in ["features", "ratios", "vector"]:
                    if fallback in row:
                        vec = row[fallback]
                        break
            if vec is None and isinstance(row.get("theories"), dict):
                theories = row["theories"]
                if feature_key in theories:
                    vec = theories[feature_key]
                elif len(theories) == 1:
                    only_key, only_val = next(iter(theories.items()))
                    vec = only_val
                    if "feature_key" not in meta:
                        meta["feature_key"] = only_key
                else:
                    _contract_error(f"{path}: theories no contiene '{feature_key}'.")
            if vec is None:
                _contract_error(f"{path}: falta vector '{key_vec}' en fila.")
            vec_list = _coerce_vector(vec)
            if vec_list is None:
                _contract_error(f"{path}: vector '{key_vec}' inválido.")
            ids.append(rid)
            vectors.append(vec_list)
        mat = _coerce_matrix(vectors, path, key_vec)
        if "columns" not in meta and meta.get("feature_key"):
            meta["columns"] = [f"{meta['feature_key']}_{i}" for i in range(mat.shape[1])]
        return ids, mat, meta

    _contract_error(
        f"Formato no reconocido en {path}. Se esperaba dict con ids+{key_vec.upper()} o lista de dicts con '{key_vec}'/theories."
    )
    raise SystemExit(2)

    return ids, mat, meta


def _decode_feature_names(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    try:
        values = np.asarray(raw)
    except Exception:
        return None
    if values.ndim == 0:
        return [str(values)]
    decoded = []
    for v in values.tolist():
        if isinstance(v, (bytes, bytearray)):
            decoded.append(v.decode("utf-8"))
        else:
            decoded.append(str(v))
    return decoded


def _normalize_ids(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        values = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        return [raw]
    normalized: List[Any] = []
    for v in values:
        if isinstance(v, (bytes, bytearray)):
            normalized.append(v.decode("utf-8"))
        else:
            normalized.append(v)
    return normalized


def load_feature_npz(path: Path) -> Tuple[Optional[List[Any]], np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        if "X" in data:
            mat = data["X"]
        elif "features" in data:
            mat = data["features"]
        else:
            raise ValueError(
                f"features.npz en {path} no contiene 'X' ni 'features'."
            )
        ids_raw = data["ids"] if "ids" in data else None
        feature_names_raw = data["feature_names"] if "feature_names" in data else None

    ids = _normalize_ids(ids_raw)
    meta: Dict[str, Any] = {}
    if feature_names_raw is not None:
        meta["columns"] = _decode_feature_names(feature_names_raw)
    return ids, np.asarray(mat, dtype=float), meta


def load_features_from_dictionary_h5(path: Path) -> Tuple[Optional[List[Any]], np.ndarray, Dict[str, Any]]:
    """Carga features desde dictionary.h5.

    Convención requerida:
      - Dataset: /features/X (2D array)
      - Opcional: /features/ids, /features/feature_names
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py no disponible; instala h5py para leer dictionary.h5") from exc

    with h5py.File(path, "r") as h5:
        if "features" not in h5 or not isinstance(h5["features"], h5py.Group):
            raise ValueError(
                f"{path} no contiene el grupo requerido /features. Se esperaba /features/X."
            )
        grp = h5["features"]
        if "X" not in grp:
            raise ValueError(
                f"{path} no contiene dataset /features/X. "
                "Define las features en ese dataset para usar este fallback."
            )
        mat = np.asarray(grp["X"])
        ids = _normalize_ids(grp["ids"][...]) if "ids" in grp else None
        feature_names = _decode_feature_names(grp["feature_names"][...]) if "feature_names" in grp else None

    meta: Dict[str, Any] = {}
    if feature_names is not None:
        meta["columns"] = feature_names
    return ids, np.asarray(mat, dtype=float), meta


# =============================================================================
# Kill-switch anti leakage
# =============================================================================


def check_leakage(
    Xs: np.ndarray,
    Ys: np.ndarray,
    meta_x: Optional[Dict[str, Any]] = None,
    meta_y: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Heurísticas defensivas. Devuelve dict con {ok: bool, reason: str?, stats: ...}.

    Criterios (abort):
    - Alguna columna de X y alguna de Y prácticamente idénticas tras estandarizar (|corr|>0.999).
    - Dimensiones iguales y matrices casi iguales (RMSE<1e-3) tras posible permutación trivial (limitada).

    Nota: NO prueba fuga; evita tautologías obvias.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "reason": "LEAKAGE_CHECK_NOT_EXECUTED",
        "executed": False,
        "max_abs_corr": None,
        "rmse_same_dim": None,
        "feature_key_match": None,
        "columns_match": None,
        "cross_corr_checked": False,
    }

    meta_x = meta_x or {}
    meta_y = meta_y or {}
    feature_key_x = meta_x.get("feature_key")
    feature_key_y = meta_y.get("feature_key")
    columns_x = meta_x.get("columns")
    columns_y = meta_y.get("columns")
    params = {
        "corr_threshold": 0.9999,
        "rmse_threshold": 1e-3,
        "same_dim_rmse_threshold": 1e-3,
        "ddof": 1,
        "top_k": 10,
        "epsilon": 1e-12,
    }
    out["params"] = params
    if not isinstance(Xs, np.ndarray) or not isinstance(Ys, np.ndarray):
        out["reason"] = "LEAKAGE_CHECK_MISSING_DATA"
        return out

    if Xs.ndim != 2 or Ys.ndim != 2:
        out["reason"] = "LEAKAGE_CHECK_INVALID_SHAPE"
        return out

    if Xs.shape[0] != Ys.shape[0]:
        out["reason"] = "LEAKAGE_CHECK_N_MISMATCH"
        return out

    if Xs.shape[0] < 2 or Xs.shape[1] < 1 or Ys.shape[1] < 1:
        out["reason"] = "LEAKAGE_CHECK_INSUFFICIENT_DATA"
        return out

    out["executed"] = True
    out["ok"] = True
    out["reason"] = None
    out["data_used"] = {
        "N": int(Xs.shape[0]),
        "dx": int(Xs.shape[1]),
        "dy": int(Ys.shape[1]),
        "columns_x": _summarize_columns(columns_x, max_items=50),
        "columns_y": _summarize_columns(columns_y, max_items=50),
    }
    feature_key_match = bool(feature_key_x and feature_key_y and feature_key_x == feature_key_y)
    columns_match = bool(columns_x and columns_y and columns_x == columns_y)
    out["feature_key_match"] = feature_key_match
    out["columns_match"] = columns_match

    # Correlaciones columna-columna (siempre que X/Y compartan N).
    # Importante: alta correlación NO implica fuga; sólo abortamos si hay evidencia
    # de *identidad* (columnas prácticamente iguales tras estandarizar).
    # Estandarización consistente (ddof=1) para evitar incoherencias numéricas.
    # La fórmula de correlación ya respeta el bound |corr|<=1; el clip es cinturón numérico.
    Xc = _standardize_ddof1(Xs, eps=params["epsilon"])
    Yc = _standardize_ddof1(Ys, eps=params["epsilon"])
    C = (Xc.T @ Yc) / max(1, Xs.shape[0] - 1)
    C = np.clip(C, -1.0, 1.0)
    absC = np.abs(C)
    max_abs_corr = float(np.max(absC))
    out["max_abs_corr"] = max_abs_corr
    out["cross_corr_checked"] = True

    # localizar el par más correlacionado y comprobar igualdad numérica
    i_max, j_max = np.unravel_index(int(np.argmax(absC)), absC.shape)
    pair_stats = _pair_stats(Xc, Yc, C, i_max, j_max)
    out["max_corr_pair_rmse"] = pair_stats["rmse"]
    out["i_max"] = int(i_max)
    out["j_max"] = int(j_max)
    out["pair_max"] = _pair_record(pair_stats, i_max, j_max, columns_x, columns_y)
    out["top_pairs"] = _top_k_pairs(absC, Xc, Yc, C, params["top_k"], columns_x, columns_y)

    if feature_key_match or columns_match:
        out.update(
            {
                "ok": False,
                "reason": "LEAKAGE_SUSPECTED: matching feature_key/columns indicates shared feature space.",
            }
        )
        return out

    # umbral muy estricto: casi identidad tras estandarizar
    if max_abs_corr > params["corr_threshold"] and pair_stats["rmse"] < params["rmse_threshold"]:
        out.update({
            "ok": False,
            "reason": (
                "LEAKAGE_SUSPECTED: columns nearly identical across X/Y after standardization "
                f"(i={i_max}, j={j_max}, corr={pair_stats['corr']:.6f}, rmse={pair_stats['rmse']:.3e})."
            ),
        })
        return out

    # Matrices casi iguales si dim coincide
    if Xs.shape[1] == Ys.shape[1]:
        Xc = _standardize_ddof1(Xs, eps=params["epsilon"])
        Yc = _standardize_ddof1(Ys, eps=params["epsilon"])
        rmse = float(np.sqrt(np.mean((Xc - Yc) ** 2)))
        out["rmse_same_dim"] = rmse
        if rmse < params["same_dim_rmse_threshold"]:
            out.update({"ok": False, "reason": "LEAKAGE_SUSPECTED: X and Y nearly identical after standardization."})
            return out

    return out


def _summarize_columns(columns: Optional[List[Any]], max_items: int = 50) -> Optional[Dict[str, Any]]:
    if not columns:
        return None
    if not isinstance(columns, (list, tuple)):
        return None
    cols = [str(c) for c in columns]
    summary: Dict[str, Any] = {"count": len(cols)}
    if len(cols) <= max_items:
        summary["columns"] = cols
        return summary
    payload = json.dumps(cols, sort_keys=True, ensure_ascii=False).encode("utf-8")
    summary["columns"] = cols[:max_items]
    summary["sha256"] = hashlib.sha256(payload).hexdigest()
    return summary


def _standardize_ddof1(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    data = np.asarray(mat, dtype=np.float64)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std < eps, eps, std)
    return (data - mean) / std


def _pair_stats(Xc: np.ndarray, Yc: np.ndarray, C: np.ndarray, i: int, j: int) -> Dict[str, float]:
    diff = Xc[:, i] - Yc[:, j]
    return {
        "corr": float(C[i, j]),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }


def _pair_record(
    stats: Dict[str, float],
    i: int,
    j: int,
    columns_x: Optional[List[Any]],
    columns_y: Optional[List[Any]],
) -> Dict[str, Any]:
    name_x = columns_x[i] if columns_x and i < len(columns_x) else None
    name_y = columns_y[j] if columns_y and j < len(columns_y) else None
    return {
        "i": int(i),
        "j": int(j),
        "x_col_index": int(i),
        "y_col_index": int(j),
        "column_x": name_x,
        "column_y": name_y,
        "stats": stats,
    }


def _top_k_pairs(
    absC: np.ndarray,
    Xc: np.ndarray,
    Yc: np.ndarray,
    C: np.ndarray,
    top_k: int,
    columns_x: Optional[List[Any]],
    columns_y: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    flat = absC.ravel()
    if flat.size == 0:
        return []
    k = int(min(top_k, flat.size))
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    pairs = []
    dx, dy = absC.shape
    for flat_idx in idx:
        i = int(flat_idx // dy)
        j = int(flat_idx % dy)
        stats = _pair_stats(Xc, Yc, C, i, j)
        pairs.append(_pair_record(stats, i, j, columns_x, columns_y))
    return pairs


# =============================================================================
# Métricas
# =============================================================================


def fit_cca(
    Xs: np.ndarray,
    Ys: np.ndarray,
    n_components: int,
    seed: int,
) -> Tuple[CCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # sklearn CCA es determinista dadas entradas; seed se usa para consistencia de bootstrap/permutaciones
    cca = CCA(n_components=n_components, max_iter=5000)
    cca.fit(Xs, Ys)
    Xc, Yc = cca.transform(Xs, Ys)
    # Correlaciones canónicas por componente (corrcoef directo)
    corrs_raw = []
    for i in range(n_components):
        x = Xc[:, i]
        y = Yc[:, i]
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            corrs_raw.append(float("nan"))
            continue
        corrs_raw.append(float(np.corrcoef(x, y)[0, 1]))
    corrs_raw_np = np.asarray(corrs_raw, dtype=float)
    corrs = np.clip(corrs_raw_np, -1.0, 1.0)
    return cca, Xc, Yc, corrs, corrs_raw_np


def bootstrap_axis_stability(
    Xs: np.ndarray,
    Ys: np.ndarray,
    n_components: int,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Estabilidad angular de ejes canónicos (U,V) vs fit completo."""
    rs = np.random.RandomState(seed)
    N = Xs.shape[0]
    cca0, _, _, _, _ = fit_cca(Xs, Ys, n_components, seed)
    Wx0 = cca0.x_weights_.copy()
    Wy0 = cca0.y_weights_.copy()

    def norm_cols(M: np.ndarray) -> np.ndarray:
        M2 = M.copy()
        for j in range(M2.shape[1]):
            n = np.linalg.norm(M2[:, j]) + 1e-12
            M2[:, j] /= n
        return M2

    Wx0 = norm_cols(Wx0)
    Wy0 = norm_cols(Wy0)

    angles = []
    cos_sims = []
    for _ in range(n_boot):
        idx = rs.randint(0, N, size=N)  # bootstrap con reemplazo
        cca, _, _, _, _ = fit_cca(Xs[idx], Ys[idx], n_components, seed)
        Wx = norm_cols(cca.x_weights_)
        Wy = norm_cols(cca.y_weights_)

        # similitud componente a componente (invariante a signo)
        cosx = np.abs(np.sum(Wx0 * Wx, axis=0))
        cosy = np.abs(np.sum(Wy0 * Wy, axis=0))
        cos = 0.5 * (cosx + cosy)
        cos = np.clip(cos, 0.0, 1.0)
        cos_sims.append(float(np.mean(cos)))
        # ángulo equivalente
        ang = float(np.degrees(np.arccos(np.mean(cos))))
        angles.append(ang)

    stability_score = float(np.mean(cos_sims))
    return {
        "bootstrap_samples": int(n_boot),
        "stability_score": stability_score,
        "mean_angle_deg": float(np.mean(angles)),
        "p90_angle_deg": float(np.percentile(angles, 90)),
        "angles_deg": angles[:200],  # recorte para no inflar JSON
    }


def permutation_significance(
    Xs: np.ndarray,
    Ys: np.ndarray,
    n_components: int,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    """Comparación score_true vs distribución permutada."""
    rs = np.random.RandomState(seed)

    _, _, _, corrs_true, _ = fit_cca(Xs, Ys, n_components, seed)
    score_true = float(np.nanmean(np.abs(corrs_true)))

    scores = []
    N = Xs.shape[0]
    for _ in range(n_perm):
        perm = rs.permutation(N)
        _, _, _, corrs, _ = fit_cca(Xs, Ys[perm], n_components, seed)
        scores.append(float(np.nanmean(np.abs(corrs))))

    scores_np = np.asarray(scores, dtype=float)
    med = float(np.median(scores_np))
    p95 = float(np.percentile(scores_np, 95))
    pval = float((np.sum(scores_np >= score_true) + 1.0) / (n_perm + 1.0))
    significance_ratio = float(score_true / (med + 1e-12))
    return {
        "permutation_samples": int(n_perm),
        "score_true": score_true,
        "score_perm_median": med,
        "score_perm_p95": p95,
        "p_value": pval,
        "significance_ratio": significance_ratio,
        "scores_perm": scores[:500],  # recorte para auditabilidad sin inflar
    }


def local_degeneracy_metrics(
    Xc: np.ndarray,
    Yc: np.ndarray,
    ids: List[Any],
    k_nn: int,
    global_var_trace: float,
) -> Dict[str, Any]:
    """Diagnóstico de degeneración/no-inyectividad local.

    Trabaja en espacio canónico (Xc,Yc). Para cada punto i:
      - vecinos en Yc (kNN)
      - estima Jacobiano local J por LS: DX @ J ≈ DY
      - cond(J) = smax/smin (smin clip) como índice de degeneración.
      - varX_trace en vecindario como proxy Var(X|Y).
    """
    N = Yc.shape[0]
    if N < 2:
        return {
            "k_nn": 0,
            "degeneracy_index_median": float("nan"),
            "degeneracy_index_p90": float("nan"),
            "degeneracy_index_max": float("nan"),
            "varX_trace_median": float("nan"),
            "varX_trace_p90": float("nan"),
            "varX_trace_ratio_median": float("nan"),
            "varX_trace_ratio_p90": float("nan"),
            "per_point": [],
        }
    k = int(min(max(3, k_nn), max(1, N - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(Yc)
    dists, neigh = nn.kneighbors(Yc, return_distance=True)
    # neigh incluye el propio punto en primera posición

    conds = []
    var_traces = []
    var_ratios = []
    per_point = []

    for i in range(N):
        js = neigh[i, 1:]  # excluir self
        DX = Xc[js] - Xc[i]
        DY = Yc[js] - Yc[i]

        # varX proxy
        covX = np.cov(Xc[js].T, bias=False)
        var_trace = float(np.trace(covX)) if covX.ndim == 2 else float(np.var(Xc[js]))
        var_traces.append(var_trace)
        var_ratio = float(var_trace / (global_var_trace + 1e-12))
        var_ratios.append(var_ratio)

        # estimar J via pseudo-inversa
        # DX: (k, dx), DY: (k, dy) => J: (dx, dy)
        try:
            J = np.linalg.pinv(DX) @ DY
            s = np.linalg.svd(J, compute_uv=False)
            smax = float(np.max(s))
            smin = float(np.min(s))
            smin = max(smin, 1e-12)
            cond = float(smax / smin)
        except Exception:
            cond = float("inf")

        conds.append(cond)
        per_point.append(
            {
                "id": ids[i],
                "cond_local": cond,
                "varX_trace": var_trace,
                "varX_trace_ratio": var_ratio,
                "k_nn": k,
            }
        )

    conds_np = np.asarray(conds, dtype=float)
    var_np = np.asarray(var_traces, dtype=float)
    var_ratio_np = np.asarray(var_ratios, dtype=float)

    return {
        "k_nn": k,
        "degeneracy_index_median": float(np.nanmedian(conds_np)),
        "degeneracy_index_p90": float(np.nanpercentile(conds_np, 90)),
        "degeneracy_index_max": float(np.nanmax(conds_np)),
        "varX_trace_median": float(np.nanmedian(var_np)),
        "varX_trace_p90": float(np.nanpercentile(var_np, 90)),
        "varX_trace_ratio_median": float(np.nanmedian(var_ratio_np)),
        "varX_trace_ratio_p90": float(np.nanpercentile(var_ratio_np, 90)),
        "per_point": per_point,
    }


def knn_preservation_metrics(
    A: np.ndarray,
    B: np.ndarray,
    ids: List[Any],
    k_nn: int,
) -> Dict[str, Any]:
    """kNN-preservation entre dos espacios (A,B)."""
    N = A.shape[0]
    if N < 2:
        return {
            "k_nn": 0,
            "overlap_mean": float("nan"),
            "overlap_median": float("nan"),
            "overlap_p10": float("nan"),
            "overlap_p90": float("nan"),
            "per_point": [],
        }
    k = int(min(max(3, k_nn), max(1, N - 1)))
    nn_a = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_b = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_a.fit(A)
    nn_b.fit(B)
    neigh_a = nn_a.kneighbors(A, return_distance=False)[:, 1:]
    neigh_b = nn_b.kneighbors(B, return_distance=False)[:, 1:]

    overlaps = []
    per_point = []
    for i in range(N):
        a = set(neigh_a[i])
        b = set(neigh_b[i])
        overlap = float(len(a.intersection(b)) / k) if k > 0 else 0.0
        overlaps.append(overlap)
        per_point.append({"id": ids[i], "overlap": overlap, "k_nn": k})

    overlaps_np = np.asarray(overlaps, dtype=float)
    return {
        "k_nn": k,
        "overlap_mean": float(np.mean(overlaps_np)),
        "overlap_median": float(np.median(overlaps_np)),
        "overlap_p10": float(np.percentile(overlaps_np, 10)),
        "overlap_p90": float(np.percentile(overlaps_np, 90)),
        "per_point": per_point,
    }


def split_atlas_positive_control(
    Xs: np.ndarray,
    ids: List[Any],
    n_components: int,
    k_nn: int,
    seed: int,
) -> Dict[str, Any]:
    dx = Xs.shape[1]
    if dx < 2:
        return {"status": "SKIP", "reason": "dx<2", "per_point": []}

    even_cols = np.arange(dx) % 2 == 0
    odd_cols = ~even_cols
    if even_cols.sum() < 1 or odd_cols.sum() < 1:
        return {"status": "SKIP", "reason": "split_failed", "per_point": []}

    ncomp = int(min(n_components, even_cols.sum(), odd_cols.sum()))
    if ncomp < 1:
        return {"status": "SKIP", "reason": "n_components<1", "per_point": []}

    cca, Xc_pos, Yc_pos, _, _ = fit_cca(Xs[:, even_cols], Xs[:, odd_cols], ncomp, seed)
    _ = cca  # noqa: F841 - solo para consistencia de estilo
    metrics = knn_preservation_metrics(Xc_pos, Yc_pos, ids, k_nn)
    return {
        "status": "OK",
        "n_components": ncomp,
        **metrics,
    }


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class Config:
    run: str
    run_x: str
    run_y: str
    atlas: Optional[str]
    features: Optional[str]
    atlas_feature_key: Optional[str] = None
    features_feature_key: Optional[str] = None
    features_from_h5: bool = False
    pairing_policy: str = "id"
    n_components: int = 3
    bootstrap_samples: int = 100
    permutation_samples: int = 300
    k_nn: int = 20
    seed: int = 42
    kill_switch: bool = True
    scale_floor_abs: float = 1e-8
    scale_floor_rel: float = 1e-6
    out_root: str = "runs"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="BASURIN F4-1: Bridge discovery via manifold alignment + injectivity diagnostics")
    p.add_argument("--run", required=False, type=str, help="run_id (legacy single-domain)")
    p.add_argument("--run-x", required=False, type=str, help="run_id dominio X (atlas)")
    p.add_argument("--run-y", required=False, type=str, help="run_id dominio Y (features)")
    p.add_argument("--allow-self-alignment", action="store_true", help="Permite run-x == run-y (no recomendado)")
    p.add_argument("--atlas", required=False, type=str, help="Path a atlas.json (X)")
    p.add_argument("--features", required=False, type=str, help="Path a features.json (Y)")
    p.add_argument("--atlas-feature-key", required=False, type=str, help="Feature key override para atlas (X)")
    p.add_argument("--features-feature-key", required=False, type=str, help="Feature key override para features (Y)")
    p.add_argument(
        "--features-from-h5",
        action="store_true",
        help="Permite fallback explícito a dictionary.h5 (/features/X).",
    )
    p.add_argument("--pairing-policy", default="id", choices=["id", "order"], help="Cómo parear X y Y (default: id)")
    p.add_argument("--n-components", default=3, type=int, help="CCA components (<=min(dx,dy))")
    p.add_argument("--bootstrap", default=100, type=int, dest="bootstrap_samples")
    p.add_argument("--perm", default=300, type=int, dest="permutation_samples")
    p.add_argument("--k-nn", default=20, type=int, dest="k_nn")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--no-kill-switch", action="store_true", help="Desactiva kill-switch (no recomendado)")
    p.add_argument("--scale-floor-abs", default=1e-8, type=float, help="Umbral absoluto para filtrar columnas casi constantes")
    p.add_argument("--scale-floor-rel", default=1e-6, type=float, help="Umbral relativo (a mediana) para filtrar columnas")
    p.add_argument("--out-root", default="runs", type=str, help="Directorio raíz de runs (default: runs)")
    a = p.parse_args()
    if a.run_x or a.run_y:
        if not (a.run_x and a.run_y):
            p.error("--run-x y --run-y deben usarse juntos")
        if a.run_x == a.run_y and not a.allow_self_alignment:
            p.error("run-x == run-y no permitido sin --allow-self-alignment")
        run = a.run_x
        run_x = a.run_x
        run_y = a.run_y
    else:
        if not a.run:
            p.error("--run es obligatorio si no se usan --run-x/--run-y")
        run = a.run
        run_x = a.run
        run_y = a.run
    return Config(
        run=run,
        run_x=run_x,
        run_y=run_y,
        atlas=a.atlas,
        features=a.features,
        atlas_feature_key=a.atlas_feature_key,
        features_feature_key=a.features_feature_key,
        features_from_h5=bool(a.features_from_h5),
        pairing_policy=a.pairing_policy,
        n_components=int(a.n_components),
        bootstrap_samples=int(a.bootstrap_samples),
        permutation_samples=int(a.permutation_samples),
        k_nn=int(a.k_nn),
        seed=int(a.seed),
        kill_switch=(not bool(a.no_kill_switch)),
        scale_floor_abs=float(a.scale_floor_abs),
        scale_floor_rel=float(a.scale_floor_rel),
        out_root=a.out_root,
    )


# =============================================================================
# Main
# =============================================================================


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def add_pairing_trace(per_point: List[Dict[str, Any]], pairing_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    pair_map = pairing_info.get("pair_map")
    traced = []
    for idx, row in enumerate(per_point):
        trace = pair_map[idx] if isinstance(pair_map, list) and idx < len(pair_map) else {}
        merged = dict(row)
        merged.update(
            {
                "pairing_policy": pairing_info.get("pairing_policy"),
                "paired_by": pairing_info.get("paired_by"),
                "atlas_id": trace.get("atlas_id") if isinstance(trace, dict) else None,
                "event_id": trace.get("event_id") if isinstance(trace, dict) else None,
                "row_i": trace.get("row_i", idx) if isinstance(trace, dict) else idx,
            }
        )
        traced.append(merged)
    return traced


def _leakage_summary(leakage: Dict[str, Any]) -> Dict[str, Any]:
    params = leakage.get("params") or {}
    return {
        "executed": leakage.get("executed"),
        "cross_corr_checked": leakage.get("cross_corr_checked"),
        "max_abs_corr": leakage.get("max_abs_corr"),
        "threshold_rmse": params.get("rmse_threshold"),
        "top_k": params.get("top_k"),
        "ddof": params.get("ddof"),
        "ok": leakage.get("ok"),
        "reason": leakage.get("reason"),
        "i_max": leakage.get("i_max"),
        "j_max": leakage.get("j_max"),
        "pair_max": leakage.get("pair_max"),
        "top_pairs": leakage.get("top_pairs"),
        "data_used": leakage.get("data_used"),
        "feature_key_match": leakage.get("feature_key_match"),
        "columns_match": leakage.get("columns_match"),
        "rmse_same_dim": leakage.get("rmse_same_dim"),
        "params": leakage.get("params"),
    }


def _scale_threshold(scales: np.ndarray, floor_abs: float, floor_rel: float) -> float:
    finite = np.asarray(scales, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    median = float(np.median(finite)) if finite.size else 0.0
    return float(max(floor_abs, floor_rel * median))


def _filter_by_scale(
    data: np.ndarray,
    scales: np.ndarray,
    floor_abs: float,
    floor_rel: float,
) -> Tuple[np.ndarray, List[int]]:
    threshold = _scale_threshold(scales, floor_abs, floor_rel)
    scales_arr = np.asarray(scales, dtype=float)
    drop_mask = (~np.isfinite(scales_arr)) | (scales_arr < threshold)
    dropped_idx = np.where(drop_mask)[0].astype(int).tolist()
    kept = data[:, ~drop_mask]
    return kept, dropped_idx


def reconcile_ids(
    ids_x: Optional[List[Any]],
    ids_y: Optional[List[Any]],
    meta_x: Dict[str, Any],
    meta_y: Dict[str, Any],
    atlas_path: Path,
) -> Tuple[Optional[List[Any]], Optional[List[Any]], Dict[str, Any]]:
    remap_info = {"applied": False, "reason": None}
    if ids_x is None or ids_y is None:
        return ids_x, ids_y, remap_info
    if ids_x == ids_y:
        return ids_x, ids_y, remap_info
    if len(ids_x) != len(ids_y):
        return ids_x, ids_y, remap_info

    ids_source = meta_y.get("ids_source")
    source_atlas_y = meta_y.get("source_atlas")
    source_atlas_x = meta_x.get("source_atlas")
    same_atlas = bool(source_atlas_x and source_atlas_y and str(source_atlas_x) == str(source_atlas_y))
    same_atlas = same_atlas or bool(source_atlas_y and str(source_atlas_y) == str(atlas_path))
    same_atlas = same_atlas or bool(source_atlas_x and str(source_atlas_x) == str(atlas_path))

    if ids_source == "atlas_points.json" or same_atlas:
        remap_info = {
            "applied": True,
            "reason": "ids remapped by order (shared atlas_points source)",
        }
        return ids_x, list(ids_x), remap_info

    return ids_x, ids_y, remap_info


def main() -> int:
    cfg = parse_args()

    try:
        out_root = resolve_out_root(cfg.out_root)
        validate_run_id(cfg.run, out_root)
        validate_run_id(cfg.run_x, out_root)
        validate_run_id(cfg.run_y, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if cfg.kill_switch:
        for run_id in {cfg.run_x, cfg.run_y}:
            contract_cmd = [
                sys.executable,
                str(_REPO_ROOT / "tools" / "contract_run_valid.py"),
                "--run",
                run_id,
            ]
            result = subprocess.run(contract_cmd, check=False)
            if result.returncode != 0:
                print(
                    "ERROR: contract_run_valid falló; abortando F4-1. "
                    "Usa --no-kill-switch para omitir este gate.",
                    file=sys.stderr,
                )
                return int(result.returncode) if result.returncode != 0 else 1

    stage_dir, outdir = ensure_stage_dirs(
        cfg.run, "bridge_f4_1_alignment", base_dir=out_root
    )

    run_dir_x = (out_root / cfg.run_x).resolve()
    run_dir_y = (out_root / cfg.run_y).resolve()

    def _resolve_input(
        run_dir: Path,
        value: Optional[str],
        default_rel: Path,
    ) -> Tuple[Path, bool]:
        used_default = value is None
        path = Path(value) if value is not None else (run_dir / default_rel)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        assert_within_runs(run_dir, path)
        return path, used_default

    def _resolve_features_path(run_dir: Path, run_label: str) -> Tuple[Path, str]:
        if cfg.features:
            path, _ = _resolve_input(run_dir, cfg.features, Path("dictionary") / "outputs" / "features.json")
            kind = path.suffix.lower()
            if kind == ".npz":
                return path, "npz"
            if kind in {".h5", ".hdf5"}:
                return path, "h5"
            return path, "json"

        features_json = run_dir / "features" / "outputs" / "features.json"
        legacy_features_json = run_dir / "dictionary" / "outputs" / "features.json"
        dictionary_h5 = run_dir / "dictionary" / "outputs" / "dictionary.h5"
        assert_within_runs(run_dir, features_json)
        assert_within_runs(run_dir, legacy_features_json)
        assert_within_runs(run_dir, dictionary_h5)
        if features_json.exists():
            return features_json, "json"
        if legacy_features_json.exists():
            return legacy_features_json, "json"
        if cfg.features_from_h5 and dictionary_h5.exists():
            return dictionary_h5, "h5"
        raise FileNotFoundError(
            "faltan features canónicas. "
            f"Ejecuta python tools/05_build_features_stage.py --run {run_label}"
        )

    try:
        atlas_path, _ = _resolve_input(
            run_dir_x,
            cfg.atlas,
            Path("dictionary") / "outputs" / "atlas.json",
        )
        features_path, features_kind = _resolve_features_path(run_dir_y, cfg.run_y)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not atlas_path.exists():
        print(f"ERROR: no existe atlas: {atlas_path}", file=sys.stderr)
        return 1
    if not features_path.exists():
        print(f"ERROR: no existe features: {features_path}", file=sys.stderr)
        return 1

    input_hashes = {
        "inputs/atlas": sha256_file(atlas_path),
        "inputs/features": sha256_file(features_path),
    }

    # Load
    ids_x, X, meta_x = load_feature_json(
        atlas_path,
        kind="atlas",
        feature_key_hint=cfg.atlas_feature_key,
    )
    try:
        if features_kind == "npz":
            ids_y, Y, meta_y = load_feature_npz(features_path)
        elif features_kind == "h5":
            ids_y, Y, meta_y = load_features_from_dictionary_h5(features_path)
        else:
            ids_y, Y, meta_y = load_feature_json(
                features_path,
                kind="features",
                feature_key_hint=cfg.features_feature_key,
            )
    except (ValueError, RuntimeError) as exc:
        print(
            "ERROR: no se pudieron resolver features canónicas. "
            f"{exc} Ejecuta python tools/05_build_features_stage.py --run {cfg.run_y}",
            file=sys.stderr,
        )
        return 1

    _load_sklearn()

    remap_info = {"applied": False, "reason": None}
    effective_pairing_policy = cfg.pairing_policy
    pairing_fallback_reason = None
    if cfg.pairing_policy == "id":
        ids_x, ids_y, remap_info = reconcile_ids(
            ids_x, ids_y, meta_x, meta_y, atlas_path
        )
        if remap_info.get("applied"):
            print(
                "WARNING: ids no coinciden; remapeando por orden (atlas_points compartido).",
                file=sys.stderr,
            )
        if ids_x is not None and ids_y is not None and ids_x != ids_y:
            # If lengths differ, automatically fall back to order pairing
            if len(ids_x) != len(ids_y):
                n_common = min(len(ids_x), len(ids_y))
                pairing_fallback_reason = f"N_X ({len(ids_x)}) != N_Y ({len(ids_y)}); using first {n_common} elements"
                print(
                    f"WARNING: {pairing_fallback_reason}",
                    file=sys.stderr,
                )
                effective_pairing_policy = "order"
            else:
                sample_x = [str(x) for x in ids_x[:5]]
                sample_y = [str(y) for y in ids_y[:5]]
                raise ValueError(
                    "IDs no coinciden para pairing-policy id. "
                    f"ids_x[0:5]={sample_x}, ids_y[0:5]={sample_y}. "
                    "Sugerencia: usa --pairing-policy order."
                )

    # Pair
    ids, Xp, Yp, pairing_info = pair_frames(ids_x, X, ids_y, Y, effective_pairing_policy)
    pairing_info["ids_remap"] = remap_info
    # Governance: explicitly record requested vs effective pairing policy
    pairing_info["pairing_policy_requested"] = cfg.pairing_policy
    pairing_info["pairing_policy_effective"] = effective_pairing_policy
    if pairing_fallback_reason:
        pairing_info["fallback_reason"] = pairing_fallback_reason

    # Basic dims
    N = Xp.shape[0]
    dx = Xp.shape[1]
    dy = Yp.shape[1]
    ncomp = int(min(cfg.n_components, dx, dy))
    if ncomp < 1:
        print(f"ERROR: n_components inválido (dx={dx}, dy={dy})", file=sys.stderr)
        return 1

    # Kill-switch leakage (estandariza internamente con ddof=1 para coherencia numérica)
    leakage = check_leakage(Xp, Yp, meta_x=meta_x, meta_y=meta_y)
    leakage_summary = _leakage_summary(leakage)
    should_abort = False
    abort_reason = None
    if not leakage.get("executed"):
        should_abort = True
        abort_reason = leakage.get("reason") or "LEAKAGE_CHECK_NOT_EXECUTED"
    elif leakage.get("feature_key_match") or leakage.get("columns_match"):
        should_abort = True
        abort_reason = leakage.get("reason") or "LEAKAGE_SUSPECTED"
    elif cfg.kill_switch and (not leakage.get("ok")):
        should_abort = True
        abort_reason = leakage.get("reason")

    if should_abort:
        cleaned_ok_outputs = []
        for stale_name in [
            "alignment_map.json",
            "metrics.json",
            "degeneracy_per_point.json",
            "knn_preservation_real.json",
            "knn_preservation_negative.json",
            "knn_preservation_control_positive.json",
            "degeneracy_hist.png",
            "degeneracy_scatter.png",
        ]:
            stale_path = outdir / stale_name
            if stale_path.exists():
                stale_path.unlink()
                cleaned_ok_outputs.append(stale_name)
        abort = {
            "stage": "bridge_f4_1_alignment",
            "version": __version__,
            "created": utcnow_iso(),
            "run": cfg.run,
            "status": "ABORT",
            "abort_reason": abort_reason,
            "leakage": leakage,
            "pairing": pairing_info,
            "data": {"N": N, "dx": dx, "dy": dy},
            "cleaned_ok_outputs": cleaned_ok_outputs,
            "config": asdict(cfg),
            "hashes": dict(input_hashes),
        }
        save_json(outdir / "abort_leakage.json", abort)
        outputs_coherence = {
            "aborted": True,
            "has_abort_file": True,
            "cleaned_stale_abort": False,
        }
        summary_path = write_stage_summary(stage_dir, {
            "stage": "bridge_f4_1_alignment",
            "version": __version__,
            "created": abort["created"],
            "run": cfg.run,
            "status": "ABORT",
            "abort_reason": abort_reason,
            "config": asdict(cfg),
            "pairing": pairing_info,
            "data": {"N": N, "dx": dx, "dy": dy, "meta_atlas": meta_x, "meta_ringdown": meta_y},
            "leakage_check": leakage_summary,
            "outputs_coherence": outputs_coherence,
            "hashes": {
                **input_hashes,
                "outputs/abort_leakage.json": sha256_file(outdir / "abort_leakage.json"),
            }
        })
        abort["hashes"]["outputs/abort_leakage.json"] = sha256_file(
            outdir / "abort_leakage.json"
        )
        write_manifest(
            stage_dir,
            {
                "abort_leakage": outdir / "abort_leakage.json",
                "summary": summary_path,
            },
            extra={
                "version": __version__,
                "status": "ABORT",
                "inputs": {
                    "atlas": str(atlas_path),
                    "features": str(features_path),
                    "features_kind": features_kind,
                    "run_x": cfg.run_x,
                    "run_y": cfg.run_y,
                },
                "outputs_coherence": outputs_coherence,
                "leakage_check": leakage_summary,
                "hashes": abort["hashes"],
            },
        )
        print(f"ABORT: {abort_reason}", file=sys.stderr)
        return 2

    # Standardize
    sx = StandardScaler(with_mean=True, with_std=True)
    sy = StandardScaler(with_mean=True, with_std=True)
    Xs = sx.fit_transform(Xp)
    Ys = sy.fit_transform(Yp)
    x_scale_raw = np.std(Xp, axis=0, ddof=0)
    y_scale_raw = np.std(Yp, axis=0, ddof=0)
    Xs_used, x_dropped_idx = _filter_by_scale(
        Xs, x_scale_raw, cfg.scale_floor_abs, cfg.scale_floor_rel
    )
    Ys_used, y_dropped_idx = _filter_by_scale(
        Ys, y_scale_raw, cfg.scale_floor_abs, cfg.scale_floor_rel
    )
    dx_used = int(Xs_used.shape[1])
    dy_used = int(Ys_used.shape[1])
    ncomp = int(min(cfg.n_components, dx_used, dy_used))
    if ncomp < 1:
        print(
            "ERROR: n_components inválido tras filtrar columnas casi constantes "
            f"(dx_used={dx_used}, dy_used={dy_used})",
            file=sys.stderr,
        )
        return 1
    abort_path = outdir / "abort_leakage.json"
    cleaned_stale_abort = False
    if abort_path.exists():
        abort_path.unlink()
        cleaned_stale_abort = True

    # Fit CCA + metrics
    cca, Xc, Yc, corrs, corrs_raw = fit_cca(Xs_used, Ys_used, ncomp, cfg.seed)
    boot = bootstrap_axis_stability(Xs_used, Ys_used, ncomp, cfg.bootstrap_samples, cfg.seed)
    perm = permutation_significance(Xs_used, Ys_used, ncomp, cfg.permutation_samples, cfg.seed)
    if Xc.shape[0] > 1:
        cov = np.cov(Xc.T, bias=False)
        # np.cov devuelve escalar si d==1; traza de escalar = escalar
        global_var_trace = float(np.trace(cov)) if getattr(cov, "ndim", 0) >= 2 else float(cov)
    else:
        global_var_trace = float(np.var(Xc))
    deg = local_degeneracy_metrics(Xc, Yc, ids, cfg.k_nn, global_var_trace)
    per_point = add_pairing_trace(deg["per_point"], pairing_info)

    # kNN preservation (real vs negativo)
    knn_real = knn_preservation_metrics(Xc, Yc, ids, cfg.k_nn)
    rs = np.random.RandomState(cfg.seed)
    perm_idx = rs.permutation(N)
    knn_neg = knn_preservation_metrics(Xc, Yc[perm_idx], ids, cfg.k_nn)
    knn_ratio = float(knn_real["overlap_mean"] / (knn_neg["overlap_mean"] + 1e-12))

    # control positivo (split atlas)
    control_pos = split_atlas_positive_control(Xs_used, ids, ncomp, cfg.k_nn, cfg.seed)

    # Save alignment map (auditable)
    cca_validation = {
        "status": "OK",
        "message": None,
    }
    raw_out_of_bounds = np.isfinite(corrs_raw) & (np.abs(corrs_raw) > (1.0 + 1e-6))
    if np.any(raw_out_of_bounds):
        cca_validation = {
            "status": "FAIL",
            "message": "cca_out_of_bounds_raw",
        }

    data_used = {
        "x_dim_original": int(dx),
        "y_dim_original": int(dy),
        "x_dim_used": int(dx_used),
        "y_dim_used": int(dy_used),
        "x_dropped_idx": x_dropped_idx,
        "y_dropped_idx": y_dropped_idx,
        "scale_floor_abs": float(cfg.scale_floor_abs),
        "scale_floor_rel": float(cfg.scale_floor_rel),
    }
    alignment = {
        "stage": "bridge_f4_1_alignment",
        "version": __version__,
        "created": utcnow_iso(),
        "run": cfg.run,
        "pairing": pairing_info,
        "standardization": {
            "x_mean": sx.mean_.tolist(),
            "x_scale": sx.scale_.tolist(),
            "y_mean": sy.mean_.tolist(),
            "y_scale": sy.scale_.tolist(),
        },
        "cca": {
            "n_components": ncomp,
            "canonical_corrs_raw": corrs_raw.tolist(),
            "canonical_corrs": corrs.tolist(),
            "canonical_corrs_method": "corrcoef(U[:,k],V[:,k]) then clip",
            "x_weights": cca.x_weights_.tolist(),
            "y_weights": cca.y_weights_.tolist(),
            "x_loadings": getattr(cca, "x_loadings_", None).tolist() if getattr(cca, "x_loadings_", None) is not None else None,
            "y_loadings": getattr(cca, "y_loadings_", None).tolist() if getattr(cca, "y_loadings_", None) is not None else None,
            "data_used": data_used,
            "validation": cca_validation,
        },
        "leakage_check": leakage,
    }
    save_json(outdir / "alignment_map.json", alignment)

    # Save per-point degeneracy
    save_json(outdir / "degeneracy_per_point.json", per_point)

    # Save kNN preservation raw
    save_json(outdir / "knn_preservation_real.json", knn_real)
    save_json(outdir / "knn_preservation_negative.json", knn_neg)
    save_json(outdir / "knn_preservation_control_positive.json", control_pos)

    # Save metrics bundle
    metrics = {
        "stage": "bridge_f4_1_alignment",
        "version": __version__,
        "created": utcnow_iso(),
        "run": cfg.run,
        "results": {
            "canonical_corr_mean": float(np.nanmean(np.abs(corrs))),
            "canonical_corrs": corrs.tolist(),
            "canonical_corrs_raw": corrs_raw.tolist(),
            "cca_validation": cca_validation,
            "stability_score": float(boot["stability_score"]),
            "mean_axis_angle_deg": float(boot["mean_angle_deg"]),
            "significance_ratio": float(perm["significance_ratio"]),
            "p_value": float(perm["p_value"]),
            "degeneracy_index_median": float(deg["degeneracy_index_median"]),
            "degeneracy_index_p90": float(deg["degeneracy_index_p90"]),
            "varX_trace_median": float(deg["varX_trace_median"]),
            "varX_trace_ratio_median": float(deg["varX_trace_ratio_median"]),
            "knn_preservation_mean": float(knn_real["overlap_mean"]),
            "knn_preservation_negative_mean": float(knn_neg["overlap_mean"]),
            "knn_preservation_ratio": float(knn_ratio),
            "control_positive_status": control_pos.get("status"),
            "control_positive_overlap_mean": float(control_pos.get("overlap_mean", float("nan"))),
            "k_nn": int(cfg.k_nn),
        },
        "bootstrap": boot,
        "permutation": perm,
        "degeneracy": {k: v for k, v in deg.items() if k != "per_point"},
        "structure_preservation": {
            "real": {k: v for k, v in knn_real.items() if k != "per_point"},
            "negative": {k: v for k, v in knn_neg.items() if k != "per_point"},
            "ratio": knn_ratio,
        },
        "control_positive": {k: v for k, v in control_pos.items() if k != "per_point"},
        "pairing": pairing_info,
        "data": {"N": N, "dx": dx, "dy": dy, "meta_atlas": meta_x, "meta_ringdown": meta_y},
        "config": asdict(cfg),
    }
    save_json(outdir / "metrics.json", metrics)

    # Optional plots (best effort)
    plot_files = {}
    try:
        import matplotlib.pyplot as plt

        conds = np.array([r["cond_local"] for r in per_point], dtype=float)
        # hist
        plt.figure()
        plt.hist(np.log10(np.clip(conds, 1e-12, 1e12)), bins=50)
        plt.xlabel("log10(cond_local)")
        plt.ylabel("count")
        plt.title("Degeneracy (local condition number)")
        fn = outdir / "degeneracy_hist.png"
        plt.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files["degeneracy_hist.png"] = "degeneracy_hist.png"

        # scatter in canonical space
        if Yc.shape[1] >= 2:
            plt.figure()
            plt.scatter(Yc[:, 0], Yc[:, 1], c=np.log10(np.clip(conds, 1e-12, 1e12)), s=12)
            plt.xlabel("Y canonical 1")
            plt.ylabel("Y canonical 2")
            plt.title("Y canonical space colored by log10(cond_local)")
            fn2 = outdir / "degeneracy_scatter.png"
            plt.savefig(fn2, dpi=150, bbox_inches="tight")
            plt.close()
            plot_files["degeneracy_scatter.png"] = "degeneracy_scatter.png"
    except Exception as e:
        # no fail: plots son opcionales
        plot_files["plots_error"] = str(e)

    outputs_coherence = {
        "aborted": False,
        "has_abort_file": abort_path.exists(),
        "cleaned_stale_abort": cleaned_stale_abort,
    }

    # stage_summary
    summary = {
        "stage": "bridge_f4_1_alignment",
        "version": __version__,
        "created": utcnow_iso(),
        "run": cfg.run,
        "status": "OK",
        "config": asdict(cfg),
        "pairing": pairing_info,
        "data": {"N": N, "dx": dx, "dy": dy, "meta_atlas": meta_x, "meta_ringdown": meta_y},
        "results": metrics["results"],
        "leakage_check": leakage_summary,
        "outputs_coherence": outputs_coherence,
        "hashes": {
            **input_hashes,
            "outputs/alignment_map.json": sha256_file(outdir / "alignment_map.json"),
            "outputs/metrics.json": sha256_file(outdir / "metrics.json"),
            "outputs/degeneracy_per_point.json": sha256_file(outdir / "degeneracy_per_point.json"),
            "outputs/knn_preservation_real.json": sha256_file(outdir / "knn_preservation_real.json"),
            "outputs/knn_preservation_negative.json": sha256_file(outdir / "knn_preservation_negative.json"),
            "outputs/knn_preservation_control_positive.json": sha256_file(outdir / "knn_preservation_control_positive.json"),
        },
    }
    # hash plots if exist
    for png in ["degeneracy_hist.png", "degeneracy_scatter.png"]:
        p = outdir / png
        if p.exists():
            summary["hashes"][f"outputs/{png}"] = sha256_file(p)
    # Governance: record plots error if matplotlib unavailable
    if "plots_error" in plot_files:
        summary["plots_error"] = plot_files["plots_error"]

    write_stage_summary(stage_dir, summary)

    # manifest
    manifest_artifacts: Dict[str, Path] = {
        "alignment_map": outdir / "alignment_map.json",
        "metrics": outdir / "metrics.json",
        "degeneracy_per_point": outdir / "degeneracy_per_point.json",
        "knn_preservation_real": outdir / "knn_preservation_real.json",
        "knn_preservation_negative": outdir / "knn_preservation_negative.json",
        "knn_preservation_control_positive": outdir / "knn_preservation_control_positive.json",
        "summary": stage_dir / "stage_summary.json",
    }
    for key, fn in plot_files.items():
        if key == "plots_error":
            continue  # Don't treat error messages as file paths
        manifest_artifacts[Path(fn).stem] = outdir / fn
    write_manifest(
        stage_dir,
        manifest_artifacts,
        extra={
            "version": __version__,
            "inputs": {
                "atlas": str(atlas_path),
                "features": str(features_path),
                "features_kind": features_kind,
                "run_x": cfg.run_x,
                "run_y": cfg.run_y,
            },
            "outputs_coherence": outputs_coherence,
            "leakage_check": leakage_summary,
        },
    )

    # Console summary
    print("=== F4-1 bridge_f4_1_alignment ===")
    print(f"N={N} dx={Xp.shape[1]} dy={Yp.shape[1]} ncomp={ncomp}")
    print(f"canonical_corr_mean={summary['results']['canonical_corr_mean']:.4f}")
    print(f"significance_ratio={perm['significance_ratio']:.3f} (p={perm['p_value']:.4g})")
    print(f"stability_score={boot['stability_score']:.3f} (mean_angle={boot['mean_angle_deg']:.2f} deg)")
    print(f"degeneracy_index_median={deg['degeneracy_index_median']:.2f}")
    print(f"Outputs: {stage_dir}")

    return 0
    
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print("ERROR: unexpected exception in bridge entrypoint", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
