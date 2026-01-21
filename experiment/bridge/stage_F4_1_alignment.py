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
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Allow running as a script by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import ensure_stage_dirs, sha256_file, write_manifest, write_stage_summary
from experiment.bridge.pairing import pair_frames
__version__ = "0.1.0"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Parsing de inputs
# =============================================================================

def _normalize_records(obj: Any, key_vec: str) -> Tuple[Optional[List[Any]], Optional[np.ndarray]]:
    """Intenta extraer (ids, matrix) de varias formas."""
    if isinstance(obj, dict):
        # Forma {"X": ..., "ids": ...}
        if key_vec.upper() in obj and "ids" in obj:
            ids = obj["ids"]
            mat = np.asarray(obj[key_vec.upper()], dtype=float)
            return ids, mat
        if key_vec in obj and "ids" in obj:
            ids = obj["ids"]
            mat = np.asarray(obj[key_vec], dtype=float)
            return ids, mat

        # Forma {"points": [...]}, {"events": [...]}
        for k in ["points", "events", "data", "rows"]:
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]
                break

    if isinstance(obj, list):
        ids = []
        rows = []
        for r in obj:
            if not isinstance(r, dict):
                return None, None
            rid = r.get("id", r.get("uid", r.get("name")))
            vec = r.get(key_vec)
            if vec is None:
                # tolerar nombres alternativos
                vec = r.get(key_vec.upper())
            if vec is None and key_vec == "x":
                vec = r.get("features", r.get("vector"))
            if vec is None:
                return None, None
            ids.append(rid)
            rows.append(vec)
        mat = np.asarray(rows, dtype=float)
        return ids, mat

    return None, None


def _coerce_vector(vec: Any) -> Optional[List[float]]:
    if isinstance(vec, list):
        return [float(x) for x in vec]
    if isinstance(vec, dict):
        for key in ["values", "vector", "data", "features"]:
            if key in vec:
                return _coerce_vector(vec[key])
    return None


def load_feature_json(path: Path, kind: str) -> Tuple[Optional[List[Any]], np.ndarray, Dict[str, Any]]:
    """Carga JSON y devuelve ids (si existen), matriz y metadatos mínimos."""
    with open(path, "r") as f:
        obj = json.load(f)

    if kind == "atlas":
        key = "x"
    elif kind == "ringdown":
        key = "y"
    else:
        raise ValueError(kind)

    if kind == "atlas" and isinstance(obj, dict) and "points" in obj:
        ids = []
        rows = []
        for point in obj.get("points", []):
            if not isinstance(point, dict):
                continue
            rid = point.get("id", point.get("uid", point.get("name")))
            vec = point.get("features", point.get("vector", point.get("x")))
            vec_list = _coerce_vector(vec)
            if vec_list is None:
                continue
            ids.append(rid)
            rows.append(vec_list)
        if rows:
            mat = np.asarray(rows, dtype=float)
            meta = {
                "feature_key": obj.get("feature_key"),
                "source_atlas": obj.get("source_atlas"),
                "n_points": obj.get("n_points"),
            }
            if "columns" in obj:
                meta["columns"] = obj.get("columns")
            if "columns" not in meta and meta.get("feature_key"):
                meta["columns"] = [f"{meta['feature_key']}_{i}" for i in range(mat.shape[1])]
            return ids, mat, meta

    ids, mat = _normalize_records(obj, key)
    if mat is None:
        raise ValueError(
            f"Formato no reconocido en {path}. Se esperaba lista de dicts con '{key}' o dict con '{key.upper()}'/ids."  # noqa
        )

    meta: Dict[str, Any] = {}
    if isinstance(obj, dict):
        if "meta" in obj and isinstance(obj["meta"], dict):
            meta.update(obj["meta"])
        for mk in ["feature_key", "k", "dim", "source", "schema_version", "created"]:
            if mk in obj:
                meta[mk] = obj[mk]
        if "columns" in obj:
            meta["columns"] = obj["columns"]

    if "columns" not in meta and "feature_key" in meta and mat is not None:
        feature_key = meta.get("feature_key")
        if feature_key:
            meta["columns"] = [f"{feature_key}_{i}" for i in range(mat.shape[1])]

    return ids, mat, meta


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
        "ok": True,
        "reason": None,
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
    feature_key_match = bool(feature_key_x and feature_key_y and feature_key_x == feature_key_y)
    columns_match = bool(columns_x and columns_y and columns_x == columns_y)
    out["feature_key_match"] = feature_key_match
    out["columns_match"] = columns_match

    if feature_key_match or columns_match:
        out.update(
            {
                "ok": False,
                "reason": "LEAKAGE_SUSPECTED: matching feature_key/columns indicates shared feature space.",
            }
        )
        return out

    if Xs.shape[0] < 10:
        return out

    compare_cross = True
    if feature_key_x and feature_key_y and feature_key_x != feature_key_y:
        compare_cross = False
    if columns_x and columns_y and columns_x != columns_y:
        compare_cross = False

    # Correlaciones columna-columna (solo si los espacios parecen comparables).
    # Importante: alta correlación NO implica fuga; sólo abortamos si hay evidencia
    # de *identidad* (columnas prácticamente iguales tras estandarizar).
    if compare_cross:
        Xc = Xs - Xs.mean(axis=0, keepdims=True)
        Yc = Ys - Ys.mean(axis=0, keepdims=True)
        Xc /= (Xs.std(axis=0, keepdims=True) + 1e-12)
        Yc /= (Ys.std(axis=0, keepdims=True) + 1e-12)

        C = (Xc.T @ Yc) / max(1, Xs.shape[0] - 1)
        absC = np.abs(C)
        max_abs_corr = float(np.max(absC))
        out["max_abs_corr"] = max_abs_corr
        out["cross_corr_checked"] = True

        # localizar el par más correlacionado y comprobar igualdad numérica
        i_max, j_max = np.unravel_index(int(np.argmax(absC)), absC.shape)
        col_rmse = float(np.sqrt(np.mean((Xc[:, i_max] - Yc[:, j_max]) ** 2)))
        out["max_corr_pair_rmse"] = col_rmse

        # umbral muy estricto: casi identidad tras estandarizar
        if max_abs_corr > 0.9999 and col_rmse < 1e-3:
            out.update({"ok": False, "reason": "LEAKAGE_SUSPECTED: columns nearly identical across X/Y after standardization."})
            return out

    # Matrices casi iguales si dim coincide
    if Xs.shape[1] == Ys.shape[1]:
        rmse = float(np.sqrt(np.mean((Xs - Ys) ** 2)))
        out["rmse_same_dim"] = rmse
        if rmse < 1e-3:
            out.update({"ok": False, "reason": "LEAKAGE_SUSPECTED: X and Y nearly identical after standardization."})
            return out

    return out


# =============================================================================
# Métricas
# =============================================================================


def fit_cca(Xs: np.ndarray, Ys: np.ndarray, n_components: int, seed: int) -> Tuple[CCA, np.ndarray, np.ndarray, np.ndarray]:
    # sklearn CCA es determinista dadas entradas; seed se usa para consistencia de bootstrap/permutaciones
    cca = CCA(n_components=n_components, max_iter=5000)
    cca.fit(Xs, Ys)
    Xc, Yc = cca.transform(Xs, Ys)
    # Correlaciones canónicas por componente
    corrs = []
    for i in range(n_components):
        x = Xc[:, i]
        y = Yc[:, i]
        denom = (np.std(x) * np.std(y)) + 1e-12
        corrs.append(float(np.cov(x, y, bias=False)[0, 1] / denom))
    return cca, Xc, Yc, np.asarray(corrs, dtype=float)


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
    cca0, _, _, _ = fit_cca(Xs, Ys, n_components, seed)
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
        cca, _, _, _ = fit_cca(Xs[idx], Ys[idx], n_components, seed)
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

    _, _, _, corrs_true = fit_cca(Xs, Ys, n_components, seed)
    score_true = float(np.mean(np.abs(corrs_true)))

    scores = []
    N = Xs.shape[0]
    for _ in range(n_perm):
        perm = rs.permutation(N)
        _, _, _, corrs = fit_cca(Xs, Ys[perm], n_components, seed)
        scores.append(float(np.mean(np.abs(corrs))))

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

    cca, Xc_pos, Yc_pos, _ = fit_cca(Xs[:, even_cols], Xs[:, odd_cols], ncomp, seed)
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
    atlas: str
    features: str
    pairing_policy: str = "id"
    n_components: int = 3
    bootstrap_samples: int = 100
    permutation_samples: int = 300
    k_nn: int = 20
    seed: int = 42
    kill_switch: bool = True
    out_root: str = "runs"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="BASURIN F4-1: Bridge discovery via manifold alignment + injectivity diagnostics")
    p.add_argument("--run", required=True, type=str, help="run_id (carpeta bajo runs/<run>/)")
    p.add_argument("--atlas", required=True, type=str, help="Path a atlas_points.json (X)")
    p.add_argument("--features", required=True, type=str, help="Path a ringdown event_features.json (Y)")
    p.add_argument("--pairing-policy", default="id", choices=["id", "order"], help="Cómo parear X y Y (default: id)")
    p.add_argument("--n-components", default=3, type=int, help="CCA components (<=min(dx,dy))")
    p.add_argument("--bootstrap", default=100, type=int, dest="bootstrap_samples")
    p.add_argument("--perm", default=300, type=int, dest="permutation_samples")
    p.add_argument("--k-nn", default=20, type=int, dest="k_nn")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--no-kill-switch", action="store_true", help="Desactiva kill-switch (no recomendado)")
    p.add_argument("--out-root", default="runs", type=str, help="Directorio raíz de runs (default: runs)")
    a = p.parse_args()
    return Config(
        run=a.run,
        atlas=a.atlas,
        features=a.features,
        pairing_policy=a.pairing_policy,
        n_components=int(a.n_components),
        bootstrap_samples=int(a.bootstrap_samples),
        permutation_samples=int(a.permutation_samples),
        k_nn=int(a.k_nn),
        seed=int(a.seed),
        kill_switch=(not bool(a.no_kill_switch)),
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
    if not pair_map or len(pair_map) != len(per_point):
        return per_point
    traced = []
    for row, trace in zip(per_point, pair_map):
        merged = dict(row)
        merged.update(
            {
                "pairing_policy": pairing_info.get("pairing_policy"),
                "paired_by": pairing_info.get("paired_by"),
                "atlas_id": trace.get("atlas_id"),
                "event_id": trace.get("event_id"),
                "row_i": trace.get("row_i"),
            }
        )
        traced.append(merged)
    return traced


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

    stage_dir, outdir = ensure_stage_dirs(
        cfg.run, "bridge_f4_1_alignment", base_dir=Path(cfg.out_root)
    )

    atlas_path = Path(cfg.atlas)
    features_path = Path(cfg.features)
    if not atlas_path.exists():
        print(f"ERROR: no existe atlas: {atlas_path}", file=sys.stderr)
        return 1
    if not features_path.exists():
        print(f"ERROR: no existe features: {features_path}", file=sys.stderr)
        return 1

    # Load
    ids_x, X, meta_x = load_feature_json(atlas_path, kind="atlas")
    ids_y, Y, meta_y = load_feature_json(features_path, kind="ringdown")

    remap_info = {"applied": False, "reason": None}
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
            sample_x = [str(x) for x in ids_x[:5]]
            sample_y = [str(y) for y in ids_y[:5]]
            raise ValueError(
                "IDs no coinciden para pairing-policy id. "
                f"ids_x[0:5]={sample_x}, ids_y[0:5]={sample_y}. "
                "Sugerencia: usa --pairing-policy order."
            )

    # Pair
    ids, Xp, Yp, pairing_info = pair_frames(ids_x, X, ids_y, Y, cfg.pairing_policy)
    pairing_info["ids_remap"] = remap_info

    # Basic dims
    N = Xp.shape[0]
    dx = Xp.shape[1]
    dy = Yp.shape[1]
    ncomp = int(min(cfg.n_components, dx, dy))
    if ncomp < 1:
        print(f"ERROR: n_components inválido (dx={dx}, dy={dy})", file=sys.stderr)
        return 1

    # Standardize
    sx = StandardScaler(with_mean=True, with_std=True)
    sy = StandardScaler(with_mean=True, with_std=True)
    Xs = sx.fit_transform(Xp)
    Ys = sy.fit_transform(Yp)

    # Kill-switch leakage
    leakage = check_leakage(Xs, Ys, meta_x=meta_x, meta_y=meta_y)
    if cfg.kill_switch and (not leakage["ok"]):
        abort = {
            "stage": "bridge_f4_1_alignment",
            "version": __version__,
            "created": utcnow_iso(),
            "run": cfg.run,
            "status": "ABORT",
            "abort_reason": leakage["reason"],
            "leakage": leakage,
            "pairing": pairing_info,
            "data": {"N": N, "dx": dx, "dy": dy},
        }
        save_json(outdir / "abort_leakage.json", abort)
        summary_path = write_stage_summary(stage_dir, {
            "stage": "bridge_f4_1_alignment",
            "version": __version__,
            "created": abort["created"],
            "run": cfg.run,
            "status": "ABORT",
            "abort_reason": leakage["reason"],
            "config": asdict(cfg),
            "pairing": pairing_info,
            "data": {"N": N, "dx": dx, "dy": dy, "meta_atlas": meta_x, "meta_ringdown": meta_y},
            "hashes": {
                "inputs/atlas": sha256_file(atlas_path),
                "inputs/features": sha256_file(features_path),
                "outputs/abort_leakage.json": sha256_file(outdir / "abort_leakage.json"),
            }
        })
        write_manifest(
            stage_dir,
            {
                "abort_leakage": outdir / "abort_leakage.json",
                "summary": summary_path,
            },
            extra={
                "version": __version__,
                "status": "ABORT",
                "inputs": {"atlas": str(atlas_path), "features": str(features_path)},
            },
        )
        print(f"ABORT: {leakage['reason']}", file=sys.stderr)
        return 2

    # Fit CCA + metrics
    cca, Xc, Yc, corrs = fit_cca(Xs, Ys, ncomp, cfg.seed)
    boot = bootstrap_axis_stability(Xs, Ys, ncomp, cfg.bootstrap_samples, cfg.seed)
    perm = permutation_significance(Xs, Ys, ncomp, cfg.permutation_samples, cfg.seed)
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
    control_pos = split_atlas_positive_control(Xs, ids, ncomp, cfg.k_nn, cfg.seed)

    # Save alignment map (auditable)
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
            "canonical_corrs": corrs.tolist(),
            "x_weights": cca.x_weights_.tolist(),
            "y_weights": cca.y_weights_.tolist(),
            "x_loadings": getattr(cca, "x_loadings_", None).tolist() if getattr(cca, "x_loadings_", None) is not None else None,
            "y_loadings": getattr(cca, "y_loadings_", None).tolist() if getattr(cca, "y_loadings_", None) is not None else None,
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
            "canonical_corr_mean": float(np.mean(np.abs(corrs))),
            "canonical_corrs": corrs.tolist(),
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
        "hashes": {
            "inputs/atlas": sha256_file(atlas_path),
            "inputs/features": sha256_file(features_path),
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
    for _, fn in plot_files.items():
        manifest_artifacts[Path(fn).stem] = outdir / fn

    write_manifest(
        stage_dir,
        manifest_artifacts,
        extra={
            "version": __version__,
            "inputs": {
                "atlas": str(atlas_path),
                "features": str(features_path),
            },
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
