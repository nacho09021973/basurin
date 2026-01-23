#!/usr/bin/env python3
"""
Bloque B: Solver Sturm-Liouville para espectro escalar en geometría AdS.

Resuelve la ecuación de Klein-Gordon masivo en el bulk:
    (□ - m²)φ = 0

En forma Sturm-Liouville estándar:
    -d/dz[p(z)φ'] + q(z)φ = λ w(z)φ

donde λ = M_n² (masa efectiva 4D, torre KK con hard wall).

Uso:
    python 03_sturm_liouville.py --run mi_experimento [--mode sweep_delta|fixed_mass]

Salida:
    runs/<run>/spectrum/
        ├── outputs/spectrum.h5          (autovalores, autofunciones, Δ, m²L²)
        ├── manifest.json        (índice de artefactos)
        ├── stage_summary.json   (metadatos, parámetros, hashes)
        └── outputs/validation.json      (ortogonalidad, residuos)

Manual tests:
    - Dual run produce M2_N sin modo espurio:
      - M2_N[:, 0] no debe ser sistemáticamente negativo/extremo.
      - validation.json contiene dual.neumann_mode_filter.
    - Ejemplo:
      python 03_sturm_liouville.py --run <run> --dual-spectrum

Convención física:
    λ ≡ M_n² = -k² = ω² - |k⃗|²  (hard-wall, self-adjoint SL)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

DEFAULT_RESIDUAL_THRESHOLD = 1e-6
DEFAULT_RESIDUAL_FIRST_K = 3

# Residual validation policy (minimal, auditable):
# - Global max is sensitive to high modes under stiff discretizations (e.g., d=5, z_min=1e-6).
# - We therefore classify quality using both global max and max over the first K modes.

# --- Repo root / runs root (NO depender de cwd) ---
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
RUNS_ROOT = Path(os.environ.get("BASURIN_RUNS_ROOT", str(ROOT_DIR / "runs")))
RUNS_ROOT = RUNS_ROOT.expanduser()
if not RUNS_ROOT.is_absolute():
    RUNS_ROOT = (Path.cwd() / RUNS_ROOT).resolve()
else:
    RUNS_ROOT = RUNS_ROOT.resolve()

from basurin_io import (
    ensure_stage_dirs,
    resolve_geometry_path,
    write_manifest,
    write_stage_summary,
    sha256_file,
)
from experiment.geometry.geometry_from_json import (
    DEFAULT_Z_MAX,
    DEFAULT_Z_MIN,
    __version__ as GEOMETRY_COMPILER_VERSION,
    compile_geometry_numeric,
    load_geometry_json,
    write_geometry_numeric,
)

# =============================================================================
# Configuración
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Configuración del solver SL."""
    run: str
    geometry_json: str | None = None
    geometry_file: str = "ads_puro.h5"
    mode: Literal["sweep_delta", "fixed_mass"] = "sweep_delta"
    dual_spectrum: bool = False
    run_kind: Literal["geometry_pipeline"] = "geometry_pipeline"
    
    # Barrido en Δ (mode=sweep_delta)
    delta_min: float = 1.55      # > d/2 para d=3 (BF bound)
    delta_max: float = 5.5
    n_delta: int = 25
    
    # Masa fija (mode=fixed_mass)
    m2L2: float = 0.0            # m²L² para modo fijo
    
    # Parámetros del solver
    n_modes: int = 10            # Modos a calcular por cada Δ
    bc_uv: Literal["dirichlet", "neumann"] = "dirichlet"
    bc_ir: Literal["dirichlet", "neumann"] = "dirichlet"
    bc_uv_D: str | None = None
    bc_ir_D: str | None = None
    bc_uv_N: str | None = None
    bc_ir_N: str | None = None
    n_z: int = 2048
    z_min: float | None = None
    z_max: float | None = None

    # Residual validation policy
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD
    residual_first_k: int = DEFAULT_RESIDUAL_FIRST_K


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Solver Sturm-Liouville para espectro escalar AdS"
    )
    p.add_argument("--run", type=str, required=True, help="Nombre del run")
    p.add_argument(
        "--geometry-json",
        type=str,
        default=None,
        dest="geometry_json",
        help="Ruta geometry.json (default runs/<run>/geometry/outputs/geometry.json)",
    )
    p.add_argument("--geometry-file", type=str, default="ads_puro.h5",
                   dest="geometry_file", help="Archivo H5 de geometría")
    p.add_argument("--mode", type=str, default="sweep_delta",
                   choices=["sweep_delta", "fixed_mass"],
                   help="Modo de operación")
    p.add_argument("--dual-spectrum", action="store_true", dest="dual_spectrum",
                   help="Activa espectro dual Dirichlet/Neumann")
    
    # Barrido Δ
    p.add_argument("--delta-min", type=float, default=1.55, dest="delta_min")
    p.add_argument("--delta-max", type=float, default=5.5, dest="delta_max")
    p.add_argument("--n-delta", type=int, default=25, dest="n_delta")
    
    # Masa fija
    p.add_argument("--m2L2", type=float, default=0.0, help="m²L² (modo fixed_mass)")
    
    # Solver
    p.add_argument("--n-modes", type=int, default=10, dest="n_modes")
    p.add_argument("--bc-uv", type=str, default="dirichlet", dest="bc_uv",
                   choices=["dirichlet", "neumann"])
    p.add_argument("--bc-ir", type=str, default="dirichlet", dest="bc_ir",
                   choices=["dirichlet", "neumann"])
    p.add_argument("--bc-uv-D", type=str, dest="bc_uv_D",
                   choices=["dirichlet", "neumann"])
    p.add_argument("--bc-ir-D", type=str, dest="bc_ir_D",
                   choices=["dirichlet", "neumann"])
    p.add_argument("--bc-uv-N", type=str, dest="bc_uv_N",
                   choices=["dirichlet", "neumann"])
    p.add_argument("--bc-ir-N", type=str, dest="bc_ir_N",
                   choices=["dirichlet", "neumann"])
    p.add_argument("--n-z", type=int, default=2048, dest="n_z")
    p.add_argument("--z-min", type=float, default=None, dest="z_min")
    p.add_argument("--z-max", type=float, default=None, dest="z_max")

    # Residual validation controls (auditable, non-breaking defaults)
    p.add_argument(
        "--residual-threshold",
        type=float,
        default=DEFAULT_RESIDUAL_THRESHOLD,
        help="Umbral para residuals_ok (backward error). Default conserva el comportamiento legacy (1e-6).",
    )
    p.add_argument(
        "--residual-first-k",
        type=int,
        default=DEFAULT_RESIDUAL_FIRST_K,
        help="K para residual_max_first_k (clasificación PASS/WARN/FAIL). Default=3.",
    )
    
    args = p.parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items()})

    # Si Config es frozen, esto es seguro; si no, también funciona.
    object.__setattr__(cfg, "residual_threshold", float(args.residual_threshold))
    object.__setattr__(cfg, "residual_first_k", int(args.residual_first_k))
    return cfg


def validate_config(cfg: Config, d: int) -> None:
    """Valida configuración contra parámetros geométricos."""
    bf_bound = d / 2.0  # Δ_min = d/2 (BF bound saturado)
    
    if cfg.mode == "sweep_delta":
        if cfg.delta_min <= bf_bound:
            raise ValueError(f"delta_min ({cfg.delta_min}) debe ser > d/2 = {bf_bound} (BF bound)")
        if cfg.delta_max <= cfg.delta_min:
            raise ValueError("delta_max debe ser > delta_min")
        if cfg.n_delta < 2:
            raise ValueError("n_delta debe ser >= 2")
    
    if cfg.n_modes < 1:
        raise ValueError("n_modes debe ser >= 1")


def resolve_dual_defaults(cfg: Config) -> None:
    """Resuelve defaults para espectro dual respetando Config frozen."""
    if not cfg.dual_spectrum:
        return
    if cfg.bc_uv_D is None:
        object.__setattr__(cfg, "bc_uv_D", cfg.bc_uv)
    if cfg.bc_ir_D is None:
        object.__setattr__(cfg, "bc_ir_D", cfg.bc_ir)
    if cfg.bc_uv_N is None:
        object.__setattr__(cfg, "bc_uv_N", "neumann")
    if cfg.bc_ir_N is None:
        object.__setattr__(cfg, "bc_ir_N", cfg.bc_ir)
    object.__setattr__(cfg, "bc_uv", cfg.bc_uv_D)
    object.__setattr__(cfg, "bc_ir", cfg.bc_ir_D)


def resolve_sweep_defaults(cfg: Config, d: int) -> None:
    """Resuelve defaults legacy para sweep_delta respetando Config frozen."""
    if cfg.mode != "sweep_delta":
        return
    bf_bound = d / 2.0
    if cfg.delta_min == 1.55 and cfg.delta_min <= bf_bound:
        object.__setattr__(cfg, "delta_min", bf_bound + 1e-3)


# =============================================================================
# Carga de geometría
# =============================================================================
def load_geometry(h5_path: Path) -> dict:
    """Carga geometría desde H5 (compatible con 01_genera_ads_puro.py)."""
    try:
        import h5py
    except ImportError:
        print("ERROR: pip install h5py", file=sys.stderr)
        sys.exit(1)
    
    with h5py.File(h5_path, "r") as h5:
        data = {
            "z": h5["z_grid"][:],
            "A": h5["A_of_z"][:],
            "f": h5["f_of_z"][:],
            "d": int(h5.attrs["d"]),
            "L": float(h5.attrs["L"]),
            "z_min": float(h5.attrs["z_min"]),
            "z_max": float(h5.attrs["z_max"]),
            "N": int(h5.attrs["N"]),
            "family": str(h5.attrs.get("family", "unknown")),
        }
    return data


def resolve_geometry_json_path(cfg: Config) -> Path:
    if cfg.geometry_json:
        candidate = Path(cfg.geometry_json)
        return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
    return (RUNS_ROOT / cfg.run / "geometry" / "outputs" / "geometry.json").resolve()


def _resolve_geometry_numeric_config(
    cfg: Config,
    geometry_payload: dict,
) -> tuple[float, float]:
    geom_z_min = geometry_payload.get("z_min")
    geom_z_max = geometry_payload.get("z_max")

    z_min = cfg.z_min if cfg.z_min is not None else geom_z_min
    z_max = cfg.z_max if cfg.z_max is not None else geom_z_max

    if z_min is None:
        z_min = DEFAULT_Z_MIN
    if z_max is None:
        z_max = DEFAULT_Z_MAX

    return float(z_min), float(z_max)


# =============================================================================
# Funciones físicas
# =============================================================================

def delta_to_m2L2(delta: float, d: int) -> float:
    """Convierte Δ a m²L² usando la relación holográfica estándar.
    
    m²L² = Δ(Δ - d)
    
    Nota: Δ_± = d/2 ± sqrt(d²/4 + m²L²)
    """
    return delta * (delta - d)


def m2L2_to_delta_plus(m2L2: float, d: int) -> float:
    """Calcula Δ₊ (rama normalizable) desde m²L².
    
    Δ₊ = d/2 + sqrt(d²/4 + m²L²)
    """
    discriminant = (d / 2.0) ** 2 + m2L2
    if discriminant < 0:
        raise ValueError(f"m²L² = {m2L2} viola BF bound (m²L² >= -{d**2/4})")
    return d / 2.0 + np.sqrt(discriminant)


def compute_sl_coefficients(z: np.ndarray, d: int, m2L2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula coeficientes p(z), q(z), w(z) para el problema SL.
    
    La ecuación de Klein-Gordon en AdS (coordenadas Poincaré):
        φ'' - (d-1)/z φ' + M²φ - m²L²/z² φ = 0
    
    En forma Sturm-Liouville -d/dz[p φ'] + q φ = λ w φ:
        p(z) = z^{-(d-1)}
        q(z) = m²L² · z^{-(d+1)}
        w(z) = z^{-(d-1)}
        λ = M²
    """
    p = np.power(z, -(d - 1), dtype=np.float64)
    q = m2L2 * np.power(z, -(d + 1), dtype=np.float64)
    w = np.power(z, -(d - 1), dtype=np.float64)
    
    return p, q, w


# =============================================================================
# Solver Sturm-Liouville (diferencias finitas)
# =============================================================================

def build_sl_matrices(
    z: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    bc_uv: str = "dirichlet",
    bc_ir: str = "dirichlet"
) -> tuple[np.ndarray, np.ndarray]:
    """Construye matrices K y W para el problema generalizado K φ = λ W φ.
    
    Usa diferencias finitas de segundo orden con tratamiento conservativo
    del término -(pφ')'.
    
    Returns:
        K: matriz de rigidez (N_int × N_int)
        W: matriz de masa diagonal (N_int × N_int)
    
    donde N_int = N - 2 para Dirichlet en ambos extremos.
    """
    N = len(z)
    h = z[1] - z[0]  # Asumimos malla uniforme
    
    # Para Dirichlet en ambos extremos: puntos interiores [1, N-2]
    N_int = N - 2
    
    if N_int < 3:
        raise ValueError(f"Malla muy gruesa: N={N}, necesito N >= 5")
    if bc_uv == "neumann" and bc_ir == "neumann" and N_int < 5:
        raise ValueError("Neumann/Neumann requiere N_int >= 5 para estabilidad numérica.")
    
    # Matrices
    K = np.zeros((N_int, N_int), dtype=np.float64)
    W = np.zeros((N_int, N_int), dtype=np.float64)
    
    # p en puntos medios: p_{i+1/2} = (p_i + p_{i+1})/2
    p_half = 0.5 * (p[:-1] + p[1:])  # tamaño N-1
    
    h2 = h * h
    
    for i in range(N_int):
        # Índice global (en la malla original)
        ig = i + 1  # i=0 interno → punto 1 en malla global
        
        # Diagonal: contribución de -(p φ')' discretizado
        # -[p_{i+1/2}(φ_{i+1}-φ_i) - p_{i-1/2}(φ_i-φ_{i-1})] / h²
        # Para φ_i: coef = (p_{i+1/2} + p_{i-1/2}) / h²
        if i == 0 and bc_uv == "neumann":
            K[i, i] = p_half[ig] / h2 + q[ig]
        elif i == N_int - 1 and bc_ir == "neumann":
            K[i, i] = p_half[ig - 1] / h2 + q[ig]
        else:
            K[i, i] = (p_half[ig] + p_half[ig - 1]) / h2 + q[ig]
        
        # Sub/super diagonal
        if i > 0:
            K[i, i - 1] = -p_half[ig - 1] / h2
        if i < N_int - 1:
            K[i, i + 1] = -p_half[ig] / h2
        
        # Matriz de masa (diagonal)
        W[i, i] = w[ig]
    
    return K, W


def solve_sl_eigenproblem(
    K: np.ndarray,
    W: np.ndarray,
    n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resuelve el problema de autovalores generalizado K φ = λ W φ.
    
    Usa scipy.linalg.eigh para problema simétrico generalizado.
    
    Returns:
        eigenvalues: array (n_modes,) con los primeros n_modes autovalores (λ = M²)
        eigenvectors: array (N_int, n_modes) con autofunciones normalizadas
    """
    from scipy.linalg import eigh
    
    # Resolver problema generalizado
    # eigh devuelve autovalores en orden ascendente
    n_total = K.shape[0]
    n_request = min(n_modes, n_total)
    
    # subset_by_index selecciona los primeros n autovalores
    eigenvalues, eigenvectors = eigh(
        K, W,
        subset_by_index=[0, n_request - 1]
    )
    
    return eigenvalues, eigenvectors


def extend_eigenfunctions(
    eigenvectors: np.ndarray,
    N: int,
    bc_uv: str,
    bc_ir: str
) -> np.ndarray:
    """Extiende autofunciones a la malla completa (añade BCs).
    
    Para Dirichlet en ambos extremos: φ[0] = φ[N-1] = 0.
    """
    N_int, n_modes = eigenvectors.shape
    phi_full = np.zeros((N, n_modes), dtype=np.float64)
    
    # Interior
    phi_full[1:-1, :] = eigenvectors
    
    if bc_uv == "neumann":
        phi_full[0, :] = phi_full[1, :]
    if bc_ir == "neumann":
        phi_full[-1, :] = phi_full[-2, :]
    
    # BCs Dirichlet ya son 0 por construcción
    return phi_full


# =============================================================================
# Validadores
# =============================================================================

def validate_orthonormality(
    eigenvectors: np.ndarray,
    W: np.ndarray,
    z: np.ndarray
) -> dict:
    """Valida ortogonalidad con peso w(z).
    
    G_mn = ∫ dz w(z) φ_m(z) φ_n(z) ≈ h · φᵀ W φ
    
    Para autofunciones normalizadas: G ≈ I.
    """
    h = z[1] - z[0]
    n_modes = eigenvectors.shape[1]
    
    # Matriz de Gram (con peso)
    # W es diagonal, así que W @ eigenvectors = w_i * φ_i
    G = h * eigenvectors.T @ (W @ eigenvectors)
    
    # Métricas
    diag = np.diag(G)
    off_diag = G - np.diag(diag)
    
    return {
        "gram_inner_product": "weighted",
        "gram_diagonal_mean": float(np.mean(diag)),
        "gram_diagonal_std": float(np.std(diag)),
        "gram_offdiag_max": float(np.max(np.abs(off_diag))),
        "gram_offdiag_mean": float(np.mean(np.abs(off_diag))),
        "orthonormality_ok": bool(np.max(np.abs(off_diag)) < 0.01 * np.mean(diag)),
    }


def validate_residuals(
    K: np.ndarray,
    W: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    cfg: Config
) -> dict:
    """Valida residuos con backward error de Kφ = λWφ."""
    residuals = []
    residual_threshold = float(cfg.residual_threshold)
    
    for i, lam in enumerate(eigenvalues):
        phi = eigenvectors[:, i]
        Kphi = K @ phi
        Wphi = W @ phi
        r = Kphi - lam * Wphi
        denom = np.linalg.norm(Kphi) + abs(lam) * np.linalg.norm(Wphi)
        denom = max(denom, 1e-300)
        residuals.append(np.linalg.norm(r) / denom)
    
    residuals = np.array(residuals, dtype=float)

    argmax_mode = int(np.argmax(residuals)) if residuals.size else None
    argmax_value = float(residuals[argmax_mode]) if argmax_mode is not None else None

    # First-K diagnostic (downstream-relevant) targets the modes most often consumed downstream.
    k_req = int(cfg.residual_first_k)
    if k_req < 1:
        raise ValueError("--residual-first-k debe ser >= 1")
    k = int(min(k_req, residuals.size)) if residuals.size else 0
    residual_first_k = residuals[:k] if k > 0 else np.array([], dtype=float)
    residual_max_first_k = float(np.max(residual_first_k)) if residual_first_k.size else 0.0
    residuals_ok_first_k = bool(residual_max_first_k < residual_threshold) if residual_first_k.size else True

    residual_max = float(np.max(residuals)) if residuals.size else 0.0
    residuals_ok = bool(residual_max < residual_threshold) if residuals.size else True

    if residuals_ok:
        residual_status = "PASS"
    elif residuals_ok_first_k:
        residual_status = "WARN"
    else:
        residual_status = "FAIL"
    
    return {
        "residual_metric": "backward_error",
        "residual_max": residual_max,
        "residual_mean": float(np.mean(residuals)) if residuals.size else 0.0,
        "residual_per_mode": residuals.tolist(),
        "residual_threshold": residual_threshold,
        "residuals_ok": residuals_ok,
        "residual_argmax_mode": argmax_mode,
        "residual_argmax_value": argmax_value,

        # New, non-breaking fields:
        "residual_first_k": k,
        "residual_max_first_k": residual_max_first_k,
        "residuals_ok_first_k": residuals_ok_first_k,
        "residual_status": residual_status,
    }


def compute_spectral_separation(
    M2_D: np.ndarray,
    M2_N: np.ndarray,
    deltas: list[float],
    n_modes: int
) -> dict:
    """Calcula diagnóstico de separación entre espectros D y N."""
    k = min(5, n_modes)
    per_delta = []
    rel_diffs = []
    
    for idx, delta in enumerate(deltas):
        m2d = M2_D[idx, :k]
        m2n = M2_N[idx, :k]
        denom = np.maximum(np.abs(m2d), 1e-12)
        sep = np.abs((m2d - m2n) / denom)
        rel_diffs.append(sep)
        ratio = None
        if abs(m2d[0]) > 1e-12:
            ratio = float(m2n[0] / m2d[0])
        per_delta.append({
            "delta": float(delta),
            "rel_diff_p50": float(np.percentile(sep, 50)),
            "rel_diff_p90": float(np.percentile(sep, 90)),
            "rel_diff_p99": float(np.percentile(sep, 99)),
            "ratio_M2_0": ratio,
        })
    
    all_rel = np.concatenate(rel_diffs) if rel_diffs else np.array([])
    global_stats = {
        "rel_diff_p50": float(np.percentile(all_rel, 50)) if all_rel.size else 0.0,
        "rel_diff_p90": float(np.percentile(all_rel, 90)) if all_rel.size else 0.0,
        "rel_diff_p99": float(np.percentile(all_rel, 99)) if all_rel.size else 0.0,
    }
    
    return {
        "k": k,
        "per_delta": per_delta,
        "global": global_stats,
    }


def filter_neumann_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_modes: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Descarta modo 0 Neumann y conserva n_modes modos válidos."""
    min_before = float(np.min(eigenvalues))
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]
    if len(eigenvalues) < n_modes:
        raise ValueError("Canal Neumann: insuficientes modos tras descartar el modo 0.")
    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]
    min_after = float(np.min(eigenvalues))
    return eigenvalues, eigenvectors, min_before, min_after


def write_outputs(
    cfg: Config,
    geo: dict,
    results: list[dict],
    validation: dict,
    geometry_path: str,
    geometry_sha256: str,
    geometry_resolution: str,
    input_geometry_absolute: str | None,
    geometry_numeric_path: Path | None = None,
    geometry_numeric_sha256: str | None = None,
    results_N: list[dict] | None = None
) -> dict:
    """Escribe todos los outputs del Bloque B.

    Contrato IO BASURIN:
      runs/<run>/spectrum/
        - manifest.json
        - stage_summary.json
        - outputs/
            - spectrum.h5
            - validation.json
    """
    import h5py

    stage_dir, outputs_dir = ensure_stage_dirs(cfg.run, "spectrum", base_dir=RUNS_ROOT)
    
    # --- spectrum.h5 ---
    h5_path = outputs_dir / "spectrum.h5"
    
    n_delta = len(results)
    n_modes = cfg.n_modes
    N = geo["N"]
    
    with h5py.File(h5_path, "w") as h5:
        # Coordenada z (de la geometría)
        h5.create_dataset("z_grid", data=geo["z"])
        
        # Arrays principales
        h5.create_dataset("delta_uv", data=np.array([r["delta"] for r in results]))
        h5.create_dataset("m2L2", data=np.array([r["m2L2"] for r in results]))
        
        # Autovalores: shape (n_delta, n_modes)
        M2_all = np.array([r["M2"] for r in results])
        
        # Autofunciones: shape (n_delta, N, n_modes)
        phi_all = np.array([r["phi"] for r in results])
        
        if cfg.dual_spectrum and results_N is not None:
            M2_all_N = np.array([r["M2"] for r in results_N])
            phi_all_N = np.array([r["phi"] for r in results_N])
            h5.create_dataset("M2_D", data=M2_all)
            h5.create_dataset("phi_D", data=phi_all)
            h5.create_dataset("M2_N", data=M2_all_N)
            h5.create_dataset("phi_N", data=phi_all_N)
            h5.create_dataset("M2", data=M2_all)
            h5.create_dataset("phi", data=phi_all)
        else:
            h5.create_dataset("M2", data=M2_all)
            h5.create_dataset("phi", data=phi_all)
        
        # Metadatos
        h5.attrs["lambda_definition"] = "M_n^2 = -k^2 = omega^2 - |k_vec|^2 (hard-wall, self-adjoint SL)"
        h5.attrs["d"] = geo["d"]
        h5.attrs["L"] = geo["L"]
        h5.attrs["n_delta"] = n_delta
        h5.attrs["n_modes"] = n_modes
        h5.attrs["bc_uv"] = cfg.bc_uv
        h5.attrs["bc_ir"] = cfg.bc_ir
        h5.attrs["dual_spectrum"] = bool(cfg.dual_spectrum)
        if cfg.dual_spectrum:
            h5.attrs["n_modes_D"] = n_modes
            h5.attrs["n_modes_N"] = n_modes
            h5.attrs["bc_uv_D"] = cfg.bc_uv_D
            h5.attrs["bc_ir_D"] = cfg.bc_ir_D
            h5.attrs["bc_uv_N"] = cfg.bc_uv_N
            h5.attrs["bc_ir_N"] = cfg.bc_ir_N
        h5.attrs["geometry_source"] = geometry_path
        h5.attrs["created"] = datetime.now(timezone.utc).isoformat()
    
    # --- validation.json ---
    val_path = outputs_dir / "validation.json"
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    
    # --- stage_summary.json ---
    inputs_payload = {
        "geometry_path": geometry_path,
        "geometry_sha256": geometry_sha256,
        "geometry_resolution": geometry_resolution,
    }
    hashes_payload = {
        "outputs/spectrum.h5": sha256_file(h5_path),
        "outputs/validation.json": sha256_file(val_path),
    }
    if geometry_numeric_path and geometry_numeric_sha256:
        inputs_payload["geometry_numeric_path"] = str(
            geometry_numeric_path.relative_to(stage_dir)
        )
        inputs_payload["geometry_numeric_sha256"] = geometry_numeric_sha256
        hashes_payload[str(geometry_numeric_path.relative_to(stage_dir))] = geometry_numeric_sha256

    summary = {
        "stage": "spectrum",
        "stage_legacy": "03_sturm_liouville",
        "run": cfg.run,
        "version": "1.1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "inputs": inputs_payload,
        "geometry": {
            "d": geo["d"],
            "L": geo["L"],
            "N": geo["N"],
            "z_min": geo["z_min"],
            "z_max": geo["z_max"],
            "family": geo["family"],
        },
        "results": {
            "n_delta": n_delta,
            "n_modes": n_modes,
            "delta_range": [results[0]["delta"], results[-1]["delta"]],
            "M2_range": [float(M2_all.min()), float(M2_all.max())],
        },
        "validation_summary": {
            "orthonormality_ok": validation["orthonormality"]["orthonormality_ok"],
            "residuals_ok": validation["residuals"]["residuals_ok"],
            "residual_metric": validation["residuals"]["residual_metric"],
            "residual_max_global": validation["residuals"]["residual_max_global"],
            "residual_argmax_mode_global": validation["residuals"]["residual_argmax_mode_global"],
            "residual_threshold": validation["residuals"]["residual_threshold"],
        },
        "hashes": hashes_payload,
    }
    summary_path = write_stage_summary(stage_dir, summary)
    
    # --- manifest.json ---
    manifest_artifacts = {
        "spectrum": h5_path,
        "validation": val_path,
        "summary": summary_path,
    }
    if geometry_numeric_path:
        manifest_artifacts["geometry_numeric"] = geometry_numeric_path
    manifest_path = write_manifest(
        stage_dir,
        manifest_artifacts,
        extra={
            "version": "1.1.0",
            "stage_legacy": "03_sturm_liouville",
            "input_geometry": geometry_path,
            "input_geometry_sha256": geometry_sha256,
            **(
                {"input_geometry_absolute": input_geometry_absolute}
                if input_geometry_absolute
                else {}
            ),
        },
    )
    
    return {
        "spectrum": h5_path,
        "validation": val_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def solve_channel(
    deltas: list[float],
    m2L2_list: list[float],
    z: np.ndarray,
    d: int,
    cfg: Config,
    bc_uv: str,
    bc_ir: str,
    label: str,
    apply_neumann_filter: bool = False
) -> tuple[list[dict], list[dict], list[dict], dict | None]:
    """Resuelve espectro para un canal (D o N)."""
    results = []
    all_ortho = []
    all_resid = []
    min_before_list = []
    min_after_list = []
    
    for i, (delta, m2L2) in enumerate(zip(deltas, m2L2_list)):
        # Coeficientes SL
        p, q, w = compute_sl_coefficients(z, d, m2L2)
        
        # Construir matrices
        K, W = build_sl_matrices(z, p, q, w, bc_uv, bc_ir)
        
        # Resolver
        n_modes_requested = cfg.n_modes + 1 if apply_neumann_filter else cfg.n_modes
        eigenvalues, eigenvectors = solve_sl_eigenproblem(K, W, n_modes_requested)
        if apply_neumann_filter:
            eigenvalues, eigenvectors, min_before, min_after = filter_neumann_modes(
                eigenvalues, eigenvectors, cfg.n_modes
            )
            min_before_list.append(min_before)
            min_after_list.append(min_after)
        
        # Extender a malla completa
        phi_full = extend_eigenfunctions(eigenvectors, len(z), bc_uv, bc_ir)
        
        # Validar
        ortho = validate_orthonormality(eigenvectors, W, z[1:-1])
        resid = validate_residuals(K, W, eigenvalues, eigenvectors, cfg=cfg)
        
        all_ortho.append(ortho)
        all_resid.append(resid)
        
        results.append({
            "delta": float(delta),
            "m2L2": float(m2L2),
            "M2": eigenvalues,
            "phi": phi_full,
        })
        
        # Progreso
        status = "✓" if ortho["orthonormality_ok"] and resid["residuals_ok"] else "⚠"
        msg = (f"  [{i+1:3d}/{len(deltas)}] ({label}) Δ={delta:.3f}, m²L²={m2L2:.3f}, "
               f"M²₀={eigenvalues[0]:.4f} {status}")
        if status == "⚠":
            msg += (f", rmax={resid['residual_argmax_value']:.2e} "
                    f"(modo {resid['residual_argmax_mode'] + 1})")
        print(msg)
    
    filter_audit = None
    if apply_neumann_filter:
        filter_audit = {
            "min_before_per_delta": min_before_list,
            "min_after_per_delta": min_after_list,
        }
    return results, all_ortho, all_resid, filter_audit


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    cfg = parse_args()
    
    # Cargar geometría
    geo = None
    geometry_path = ""
    geometry_sha256 = ""
    geometry_resolution = ""
    input_geometry_absolute = None
    geometry_numeric_sha256 = None
    geometry_numeric_path = None
    geometry_source_path = None

    geometry_json_path = resolve_geometry_json_path(cfg)
    geometry_file_path = None
    if cfg.geometry_file:
        try:
            geometry_file_path, geometry_path, input_geometry_absolute, geometry_resolution = resolve_geometry_path(
                cfg.run, cfg.geometry_file, base_dir=RUNS_ROOT
            )
        except (ValueError, FileNotFoundError):
            geometry_file_path = None

    if geometry_file_path is not None and geometry_file_path.exists():
        geometry_sha256 = sha256_file(geometry_file_path)
        geo = load_geometry(geometry_file_path)
        geometry_source_path = geometry_file_path
    elif geometry_json_path.exists():
        try:
            geom_decl = load_geometry_json(geometry_json_path)
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        z_min, z_max = _resolve_geometry_numeric_config(cfg, geom_decl)
        compiled = compile_geometry_numeric(geom_decl, cfg.n_z, z_min, z_max)
        run_dir = (RUNS_ROOT / cfg.run).resolve()
        try:
            geometry_path = str(geometry_json_path.relative_to(run_dir))
        except ValueError:
            print(
                "ERROR: geometry.json debe vivir dentro de runs/<run>/geometry/outputs/",
                file=sys.stderr,
            )
            return 1
        stage_dir, outputs_dir = ensure_stage_dirs(cfg.run, "spectrum", base_dir=RUNS_ROOT)
        inputs_dir = stage_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        geometry_numeric_path = inputs_dir / "geometry_numeric.json"
        geometry_sha256 = sha256_file(geometry_json_path)
        geometry_resolution = "geometry_json"
        compiled_payload = {
            "source_geometry_path": geometry_path,
            "geometry_sha256": geometry_sha256,
            "compiler_version": GEOMETRY_COMPILER_VERSION,
            "numeric_config": {
                "n_z": cfg.n_z,
                "z_min": z_min,
                "z_max": z_max,
            },
            "geometry": geom_decl["raw"],
            "numeric": compiled,
        }
        geometry_numeric_sha256 = write_geometry_numeric(geometry_numeric_path, compiled_payload)
        geo = {
            "z": np.array(compiled["z"], dtype=np.float64),
            "A": np.array(compiled["A"], dtype=np.float64),
            "f": np.array(compiled["f"], dtype=np.float64),
            "d": compiled["d"],
            "L": compiled["L"],
            "z_min": compiled["z_min"],
            "z_max": compiled["z_max"],
            "N": compiled["N"],
            "family": compiled["family"],
        }
        geometry_source_path = geometry_json_path
    else:
        print(
            "ERROR: geometría no encontrada. Proporciona --geometry-file (H5 existente) "
            "o un geometry.json en runs/<run>/geometry/outputs/geometry.json.",
            file=sys.stderr,
        )
        return 1

    d, L = geo["d"], geo["L"]
    z = geo["z"]
    
    print(f"Geometría cargada: {geometry_source_path}")
    print(f"  d={d}, L={L}, N={geo['N']}, z∈[{geo['z_min']}, {geo['z_max']}]")
    print()
    
    # Validar config
    resolve_sweep_defaults(cfg, d)
    validate_config(cfg, d)
    resolve_dual_defaults(cfg)
    
    # Construir lista de (Δ, m²L²) a procesar
    if cfg.mode == "sweep_delta":
        deltas = np.linspace(cfg.delta_min, cfg.delta_max, cfg.n_delta)
        m2L2_list = [delta_to_m2L2(delta, d) for delta in deltas]
        print(f"Modo: sweep_delta")
        print(f"  Δ ∈ [{cfg.delta_min}, {cfg.delta_max}], n_delta={cfg.n_delta}")
    else:
        m2L2_list = [cfg.m2L2]
        deltas = [m2L2_to_delta_plus(cfg.m2L2, d)]
        print(f"Modo: fixed_mass")
        print(f"  m²L² = {cfg.m2L2}, Δ₊ = {deltas[0]:.4f}")
    
    print(f"  n_modes = {cfg.n_modes}")
    print()
    
    # --- Resolver para cada Δ ---
    results_N = None
    all_ortho_N = None
    all_resid_N = None
    neumann_filter_audit = None
    
    if cfg.dual_spectrum:
        results, all_ortho, all_resid, _ = solve_channel(
            deltas, m2L2_list, z, d, cfg, cfg.bc_uv_D, cfg.bc_ir_D, "D"
        )
        results_N, all_ortho_N, all_resid_N, neumann_filter_audit = solve_channel(
            deltas, m2L2_list, z, d, cfg, cfg.bc_uv_N, cfg.bc_ir_N, "N",
            apply_neumann_filter=True
        )
    else:
        results, all_ortho, all_resid, _ = solve_channel(
            deltas, m2L2_list, z, d, cfg, cfg.bc_uv, cfg.bc_ir, "L"
        )
    
    print()
    
    # --- Validación agregada ---
    residual_metric = all_resid[0]["residual_metric"] if all_resid else "backward_error"
    residual_threshold = all_resid[0]["residual_threshold"] if all_resid else None
    _, max_resid = max(enumerate(all_resid), key=lambda item: item[1]["residual_max"])
    validation = {
        "orthonormality": {
            "orthonormality_ok": all(o["orthonormality_ok"] for o in all_ortho),
            "gram_offdiag_max_global": max(o["gram_offdiag_max"] for o in all_ortho),
            "per_delta": all_ortho,
        },
        "residuals": {
            "residual_metric": residual_metric,
            "residuals_ok": all(r["residuals_ok"] for r in all_resid),
            "residual_max_global": max(r["residual_max"] for r in all_resid),
            "residual_argmax_mode_global": max_resid.get("residual_argmax_mode"),
            "residual_threshold": residual_threshold,
            "per_delta": all_resid,
        },
    }
    
    if cfg.dual_spectrum and results_N is not None and all_ortho_N is not None and all_resid_N is not None:
        M2_all_D = np.array([r["M2"] for r in results])
        M2_all_N = np.array([r["M2"] for r in results_N])
        min_before_list = []
        min_after_list = []
        if neumann_filter_audit is not None:
            min_before_list = neumann_filter_audit.get("min_before_per_delta", [])
            min_after_list = neumann_filter_audit.get("min_after_per_delta", [])
        min_before_global = float(min(min_before_list)) if min_before_list else None
        min_after_global = float(min(min_after_list)) if min_after_list else None
        validation["dual"] = {
            "orthonormality_N": {
                "orthonormality_ok": all(o["orthonormality_ok"] for o in all_ortho_N),
                "gram_offdiag_max_global": max(o["gram_offdiag_max"] for o in all_ortho_N),
                "per_delta": all_ortho_N,
            },
            "residuals_N": {
                "residual_metric": all_resid_N[0]["residual_metric"],
                "residuals_ok": all(r["residuals_ok"] for r in all_resid_N),
                "residual_max_global": max(r["residual_max"] for r in all_resid_N),
                "residual_argmax_mode_global": max(
                    all_resid_N, key=lambda item: item["residual_max"]
                ).get("residual_argmax_mode"),
                "residual_threshold": all_resid_N[0]["residual_threshold"],
                "per_delta": all_resid_N,
            },
            "neumann_mode_filter": {
                "dropped_modes": [0],
                "kept_from_mode": 1,
                "reason": "Neumann fundamental mode dropped (non-informative / numerically spurious in this formulation)",
                "min_M2_N_before_per_delta": min_before_list,
                "min_M2_N_after_per_delta": min_after_list,
                "min_M2_N_before_global": min_before_global,
                "min_M2_N_after_global": min_after_global,
            },
            "spectral_separation": compute_spectral_separation(
                M2_all_D, M2_all_N, list(deltas), cfg.n_modes
            ),
        }
    
    # --- Escribir outputs ---
    paths = write_outputs(
        cfg,
        geo,
        results,
        validation,
        geometry_path,
        geometry_sha256,
        geometry_resolution,
        input_geometry_absolute,
        geometry_numeric_path=geometry_numeric_path,
        geometry_numeric_sha256=geometry_numeric_sha256,
        results_N=results_N,
    )
    
    print("Outputs escritos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print()
    
    # --- Resumen ---
    ortho_ok = validation["orthonormality"]["orthonormality_ok"]
    resid_ok = validation["residuals"]["residuals_ok"]
    
    if ortho_ok and resid_ok:
        print("VERIFICADO: Ortogonalidad y residuos OK")
        return 0
    else:
        if not ortho_ok:
            print(f"ADVERTENCIA: Ortogonalidad degradada "
                  f"(max off-diag = {validation['orthonormality']['gram_offdiag_max_global']:.2e})")
        if not resid_ok:
            print(f"ADVERTENCIA: Residuos elevados "
                  f"(max = {validation['residuals']['residual_max_global']:.2e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
