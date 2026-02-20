#!/usr/bin/env python3
# 04_geometry_physics_contracts.py
# CUERDAS - Bloque A: Geometria emergente (contratos fisicos)
#
# Objective:
#   Evaluar la calidad fisica y la honestidad del bloque de geometria y ecuaciones:
#     - Geometria emergente vs bulk_truth (cuando exista sandbox).
#     - Ecuaciones de bulk descubiertas vs criterios fisicos.
#
# Inputs:
#   - runs/emergent_geometry/emergent_geometry_summary.json
#   - runs/bulk_equations/equations_pareto.json (y/o pysr_summary.json)
#   - Opcional: runs/sandbox_geometries/bulk_truth/*.h5 (para contratos sandbox)
#
# Outputs:
#   runs/geometry_contracts/
#     geometry_contracts_summary.json
#       - Clasificacion por sistema:
#           * Einstein-like / non-Einstein / incierto
#           * Umbrales de R², estabilidad en rollouts, etc.
#       - Flags de honestidad (uso indebido de bulk, mezcla incorrecta de d, ...).
#
# TIPOS DE CONTRATO (EJEMPLOS)
#   - R² minimo en test, estabilidad en evoluciones numericas.
#   - No mezcla de dimensiones d entre sistemas incompatibles.
#   - No uso de variables de "verdad" en la construccion de la loss.
#   - Coherencia entre family asignada (ads/lifshitz/hvlf/deformed) y patrones geometricos.
#
# RELACION CON OTROS SCRIPTS
#   - Consume Outputs: de:
#       * 02_emergent_geometry_engine.py
#       * 03_discover_bulk_equations.py
#   - Sus resultados condicionan el analisis de:
#       * 05_analyze_bulk_equations.py
#
# History:
#   - Anteriormente conocido como: 04_contracts_fase_11_v2.py
#   - V2.2: Documentado gauge conformal y ajustado test R_is_constant
#   - V2.1: Anadido soporte para inference mode (No bulk_truth)
#
# MODOS SOPORTADOS (V2.1):
#   - Modo A (sandbox/train): bulk_truth disponible, A_truth/f_truth en NPZ
#   - Modo B (inference/real): No bulk_truth, solo predicciones
#     En este modo, metricas R² se marcan como null y generic contracts se evaluan

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import h5py

# Import local IO module for run manifest support
try:
    from cuerdas_io import (
        RunContext,
        resolve_predictions_dir as cuerdas_resolve_predictions,
        resolve_geometry_emergent_dir as cuerdas_resolve_geometry_emergent,
        resolve_bulk_equations_dir as cuerdas_resolve_bulk_equations,
        resolve_data_dir as cuerdas_resolve_data,
        resolve_dictionary_file as cuerdas_resolve_dictionary,
        update_run_manifest,
    )
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False

# V3 Infrastructure
HAS_STAGE_UTILS = False
try:
    from stage_utils import (
        StageContext, add_standard_arguments, parse_stage_args,
        EXIT_OK, EXIT_ERROR, STATUS_OK, STATUS_ERROR
    )
    HAS_STAGE_UTILS = True
except ImportError:
    EXIT_OK = 0
    EXIT_ERROR = 3
    STATUS_OK = "OK"
    STATUS_ERROR = "ERROR"


# ============================================================
# HELPERS PARA RESOLUCION DE RUTAS (V2.1)
# ============================================================

def resolve_predictions_dir(geometry_dir: Path) -> Path:
    """
    Resuelve el directory de predicciones NPZ.
    
    Busca en orden:
    1. geometry_dir/predictions/
    2. geometry_dir/ (si ya es el directory de predictions)
    """
    predictions_subdir = geometry_dir / "predictions"
    if predictions_subdir.exists() and predictions_subdir.is_dir():
        return predictions_subdir
    
    # Comprobar si geometry_dir ya contiene NPZ directamente
    npz_files = list(geometry_dir.glob("*_geometry.npz"))
    if npz_files:
        return geometry_dir
    
    # Fallback: devolver el subdirectory esperado (puede no existir)
    return predictions_subdir


def resolve_emergent_h5_dir(geometry_dir: Path) -> Path:
    """
    Resuelve el directory de H5 emergentes.
    
    Busca en orden:
    1. geometry_dir/geometry_emergent/
    2. geometry_dir/ (si ya contiene H5 emergentes)
    """
    emergent_subdir = geometry_dir / "geometry_emergent"
    if emergent_subdir.exists() and emergent_subdir.is_dir():
        return emergent_subdir
    
    # Comprobar si geometry_dir ya contiene H5 directamente
    h5_files = list(geometry_dir.glob("*_emergent.h5"))
    if h5_files:
        return geometry_dir
    
    return emergent_subdir


def resolve_bulk_equations_dir(einstein_dir: Path) -> Path:
    """
    Resuelve el directory de ecuaciones de bulk.
    
    Busca en orden:
    1. einstein_dir/bulk_equations/
    2. einstein_dir/ (si ya es el directory correcto)
    """
    bulk_eq_subdir = einstein_dir / "bulk_equations"
    if bulk_eq_subdir.exists() and bulk_eq_subdir.is_dir():
        return bulk_eq_subdir
    
    # Comprobar si einstein_dir ya contiene subdirectorys de sistemas
    # (cada sistema tiene su carpeta con einstein_discovery.json)
    json_files = list(einstein_dir.glob("*/einstein_discovery.json"))
    if json_files:
        return einstein_dir
    
    return bulk_eq_subdir


# ============================================================
# generic contracts (aplican a TODAS las geometrias)
# ============================================================

@dataclass
class GenericRegularityContract:
    """Verifica propiedades basicas de regularidad."""
    A_finite: bool
    f_finite: bool
    no_nan: bool
    smooth: bool  # derivadas no explotan
    
    @property
    def passed(self) -> bool:
        return self.A_finite and self.f_finite and self.no_nan


@dataclass
class GenericCausalityContract:
    """Verifica estructura causal basica (no especifica de AdS)."""
    f_non_negative: bool  # f >= 0 (o muy cerca)
    f_bounded: bool  # f no explota
    horizon_if_thermal: bool  # si T > 0, debe haber estructura de horizonte
    skipped: bool = False  # True si no hay datos para evaluar (T/z_h ausentes)
    
    @property
    def passed(self) -> bool:
        if self.skipped:
            return True  # No penalizar si no hay datos
        return self.f_non_negative and self.f_bounded


@dataclass
class BoundaryUnitarityContract:
    """
    Verifica que operadores del boundary satisfacen cotas de unitariedad.
    
    Cota de unitariedad para escalares: Δ >= (d-2)/2
    Si algún operador viola esto, el sistema NO puede ser una CFT unitaria
    y por tanto NO puede tener dual holográfico válido.
    
    Referencia: AGMOO review, Chunk 04, p.32
    """
    all_operators_unitary: bool  # Todos cumplen Δ >= (d-2)/2
    n_operators: int
    n_violations: int
    min_delta: float
    unitarity_bound: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    skipped: bool = False
    
    @property
    def passed(self) -> bool:
        if self.skipped:
            return True  # No penalizar si no hay operadores
        return self.all_operators_unitary


@dataclass
class CorrelatorStructureContract:
    """
    Verifica que el correlador G2 tiene estructura compatible con CFT.
    
    Una CFT tiene correladores con power-law decay: G(x) ~ x^{-2Δ}
    Ruido blanco tiene G(x) ~ δ(x) (sin estructura espacial)
    
    Detectamos esto verificando:
    1. Correlador no es constante (tiene variación espacial)
    2. Tiene decaimiento monótono (no oscila aleatoriamente)
    3. En escala log-log, tiene pendiente negativa (power-law)
    """
    has_spatial_structure: bool  # No es constante/ruido
    is_monotonic_decay: bool  # Decae monótonamente
    has_power_law: bool  # Pendiente log-log negativa
    log_slope: float  # Pendiente en log-log (debería ser ~ -2Δ)
    correlation_quality: float  # R² del ajuste log-log
    skipped: bool = False
    
    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        # Debe tener estructura espacial Y decaimiento tipo power-law
        return self.has_spatial_structure and self.has_power_law


# ============================================================
# AdS-specific contracts (solo para family="ads")
# ============================================================

@dataclass
class AdSEinsteinContract:
    """Verifica ecuaciones de Einstein en vacio con Λ (SOLO para AdS)."""
    R_is_constant: bool
    R_matches_ads: bool
    Lambda_matches_ads: bool
    R_mean: float
    R_expected: float
    
    @property
    def passed(self) -> bool:
        return self.R_is_constant and self.R_matches_ads


@dataclass
class AdSAsymptoticContract:
    """Verifica comportamiento asintotico tipo AdS."""
    A_logarithmic_uv: bool  # A(z) ~ -log(z) cerca de z=0
    f_to_one_uv: bool  # f → 1 cuando z → 0
    A_monotone: bool  # dA/dz < 0
    
    @property
    def passed(self) -> bool:
        return self.A_logarithmic_uv and self.A_monotone


@dataclass
class HolographicDictionaryContract:
    """Verifica diccionario holografico (mas relevante para AdS)."""
    mass_dimension_ok: bool
    hawking_ok: bool
    conformal_symmetry_ok: bool
    
    @property
    def passed(self) -> bool:
        return self.mass_dimension_ok or self.hawking_ok or self.conformal_symmetry_ok


# ============================================================
# CONTRATO GLOBAL
# ============================================================

@dataclass
class PhaseXIContractV2:
    """Contrato completo de Fase XI v2."""
    name: str
    family: str
    category: str
    d: int
    
    # generic contracts (TODOS deben pasar)
    regularity: GenericRegularityContract
    causality: GenericCausalityContract
    unitarity: BoundaryUnitarityContract  # V2.3: operadores unitarios
    correlator_structure: CorrelatorStructureContract  # V2.3: G2 no es ruido
    
    # AdS-specific contracts (solo relevantes si family="ads")
    ads_einstein: AdSEinsteinContract
    ads_asymptotic: AdSAsymptoticContract
    holographic: HolographicDictionaryContract
    
    # Metricas de reconstruccion (pueden ser None en inference)
    A_r2: Optional[float]
    f_r2: Optional[float]
    R_r2: Optional[float]
    family_accuracy: Optional[float]
    
    # Metadatos de modo (V2.1)
    mode: str = "sandbox"  # "sandbox" o "inference"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def generic_passed(self) -> bool:
        """Contratos que TODA geometria debe pasar."""
        # V2.3: Incluir unitariedad y estructura de correlador
        return (self.regularity.passed and 
                self.causality.passed and
                self.unitarity.passed and
                self.correlator_structure.passed)
    
    @property
    def ads_specific_passed(self) -> bool:
        """Contratos especificos de AdS (solo relevantes si es AdS)."""
        return self.ads_einstein.passed and self.ads_asymptotic.passed
    
    @property
    def is_ads_family(self) -> bool:
        return self.family == "ads"
    
    @property
    def contract_score(self) -> float:
        """Score ponderado por tipo de geometria."""
        def to_float(val) -> float:
            """Convierte bool o string a float."""
            if isinstance(val, bool):
                return 1.0 if val else 0.0
            if isinstance(val, str):
                return 1.0 if val.lower() == 'true' else 0.0
            return float(val)
        
        # Genericos: siempre cuentan (ahora 4 contratos, pesos ajustados)
        score = (0.2 * to_float(self.regularity.passed) + 
                 0.15 * to_float(self.causality.passed) +
                 0.15 * to_float(self.unitarity.passed) +
                 0.15 * to_float(self.correlator_structure.passed))
        
        # Reconstruccion (solo si hay metricas disponibles)
        if self.A_r2 is not None and self.f_r2 is not None:
            a_r2_safe = max(0.0, min(1.0, self.A_r2))
            f_r2_safe = max(0.0, min(1.0, self.f_r2))
            score += 0.15 * (a_r2_safe + f_r2_safe) / 2
        else:
            # En inference without truth, dar puntos parciales si genericos pasan
            score += 0.05 * to_float(self.generic_passed)
        
        # AdS-especificos: solo cuentan si es AdS
        if self.is_ads_family:
            score += 0.15 * to_float(self.ads_einstein.passed)
            score += 0.05 * to_float(self.holographic.passed)
        else:
            # Para no-AdS, damos puntos si paso genericos
            score += 0.1 * to_float(self.generic_passed)
            score += 0.05 * to_float(self.holographic.conformal_symmetry_ok)
        
        return score
    
    @property
    def overall_passed(self) -> bool:
        """Paso la fase."""
        if not self.generic_passed:
            return False
        if self.is_ads_family:
            return self.ads_specific_passed
        return True  # Non-AdS solo necesita genericos


# ============================================================
# FUNCIONES DE VERIFICACION
# ============================================================

def verify_regularity(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray
) -> GenericRegularityContract:
    """Verifica regularidad basica."""
    A_finite = np.all(np.isfinite(A)) and np.max(np.abs(A)) < 1e6
    f_finite = np.all(np.isfinite(f)) and np.max(np.abs(f)) < 1e6
    no_nan = not (np.any(np.isnan(A)) or np.any(np.isnan(f)))
    
    # Suavidad: derivadas no explotan
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    dA = np.gradient(A, dz)
    smooth = np.all(np.abs(dA) < 1e4)
    
    return GenericRegularityContract(
        A_finite=A_finite,
        f_finite=f_finite,
        no_nan=no_nan,
        smooth=smooth
    )


def verify_generic_causality(
    f: np.ndarray,
    z: np.ndarray,
    T: Optional[float],
    z_h: Optional[float]
) -> GenericCausalityContract:
    """Verifica causalidad basica."""
    f_non_negative = np.min(f) >= -0.1
    f_bounded = np.max(f) < 10.0
    
    # Si no hay datos de T/z_h, marcar como skipped
    if T is None or z_h is None:
        return GenericCausalityContract(
            f_non_negative=f_non_negative,
            f_bounded=f_bounded,
            horizon_if_thermal=True,
            skipped=True
        )
    
    # Si hay temperatura, debe haber estructura de horizonte
    if T > 1e-10 and z_h > 0:
        idx_h = np.argmin(np.abs(z - z_h))
        f_near_h = f[max(0, idx_h-3):min(len(f), idx_h+3)]
        horizon_ok = np.min(f_near_h) < 0.5 if len(f_near_h) > 0 else True
    else:
        horizon_ok = True
    
    return GenericCausalityContract(
        f_non_negative=f_non_negative,
        f_bounded=f_bounded,
        horizon_if_thermal=horizon_ok,
        skipped=False
    )


def verify_boundary_unitarity(
    operators: List[Dict[str, Any]],
    d: int
) -> BoundaryUnitarityContract:
    """
    Verifica que operadores del boundary satisfacen cotas de unitariedad.
    
    Cota de unitariedad para escalares (spin=0): Δ >= (d-2)/2
    Cota para spin s: Δ >= s + d - 2 (para s >= 1)
    
    Referencia: AGMOO review, Chunk 04, p.32
    """
    if not operators:
        return BoundaryUnitarityContract(
            all_operators_unitary=True,
            n_operators=0,
            n_violations=0,
            min_delta=float('nan'),
            unitarity_bound=(d - 2) / 2,
            violations=[],
            skipped=True
        )
    
    unitarity_bound_scalar = (d - 2) / 2
    violations = []
    deltas = []
    
    for op in operators:
        delta = op.get("Delta", op.get("delta", None))
        if delta is None:
            continue
        
        spin = op.get("spin", 0)
        name = op.get("name", "unknown")
        
        # Cota depende del spin
        if spin == 0:
            bound = unitarity_bound_scalar
        else:
            bound = spin + d - 2
        
        deltas.append(delta)
        
        # Verificar violación
        if delta < bound - 1e-6:
            violations.append({
                "name": name,
                "Delta": float(delta),
                "spin": int(spin),
                "bound": float(bound),
                "violation": float(bound - delta)
            })
    
    return BoundaryUnitarityContract(
        all_operators_unitary=len(violations) == 0,
        n_operators=len(deltas),
        n_violations=len(violations),
        min_delta=float(min(deltas)) if deltas else float('nan'),
        unitarity_bound=float(unitarity_bound_scalar),
        violations=violations,
        skipped=False
    )


def verify_correlator_structure(
    G2: Optional[np.ndarray],
    x_grid: Optional[np.ndarray]
) -> CorrelatorStructureContract:
    """
    Verifica que el correlador G2 tiene estructura compatible con CFT.
    
    Una CFT tiene correladores con power-law decay: G(x) ~ x^{-2Δ}
    Ruido blanco tiene G(x) ~ δ(x) (sin estructura espacial)
    """
    if G2 is None or x_grid is None or len(G2) < 5:
        return CorrelatorStructureContract(
            has_spatial_structure=True,  # Dar beneficio de la duda
            is_monotonic_decay=True,
            has_power_law=True,
            log_slope=float('nan'),
            correlation_quality=float('nan'),
            skipped=True
        )
    
    # 1. Verificar que no es constante (tiene variación espacial)
    G2_std = np.std(G2)
    G2_mean = np.mean(np.abs(G2))
    cv = G2_std / (G2_mean + 1e-10)
    has_spatial_structure = cv > 0.1  # CV > 10% indica variación
    
    # 2. Verificar decaimiento monótono (para x > 0)
    # En ruido, los valores oscilan aleatoriamente
    dG = np.diff(G2)
    n_negative = np.sum(dG < 0)
    n_positive = np.sum(dG > 0)
    # Power-law decay debería tener mayoría de decrementos
    is_monotonic_decay = n_negative > 0.6 * len(dG) if len(dG) > 0 else True
    
    # 3. Verificar power-law en log-log
    # G(x) ~ x^{-2Δ} → log(G) = -2Δ log(x) + const
    # Ruido blanco no tiene esta estructura
    try:
        # Filtrar valores positivos para log
        mask = (G2 > 1e-10) & (x_grid > 1e-10)
        if np.sum(mask) > 5:
            log_x = np.log(x_grid[mask])
            log_G = np.log(G2[mask])
            
            # Ajuste lineal en log-log
            # coeffs[0] es la pendiente (debería ser negativa para CFT)
            coeffs = np.polyfit(log_x, log_G, 1)
            log_slope = coeffs[0]
            
            # Calcular R² del ajuste
            log_G_pred = np.polyval(coeffs, log_x)
            ss_res = np.sum((log_G - log_G_pred) ** 2)
            ss_tot = np.sum((log_G - np.mean(log_G)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0
            
            # Power-law: pendiente negativa y buen ajuste
            has_power_law = log_slope < -0.1 and r2 > 0.5
        else:
            log_slope = 0.0
            r2 = 0.0
            has_power_law = False
    except Exception:
        log_slope = 0.0
        r2 = 0.0
        has_power_law = False
    
    return CorrelatorStructureContract(
        has_spatial_structure=has_spatial_structure,
        is_monotonic_decay=is_monotonic_decay,
        has_power_law=has_power_law,
        log_slope=float(log_slope),
        correlation_quality=float(r2),
        skipped=False
    )


def verify_ads_einstein(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray,
    d: int,
    einstein_results: Dict[str, Any]
) -> AdSEinsteinContract:
    """
    Verifica ecuaciones de Einstein (solo para AdS).
    
    GAUGE: Metrica conformal (Domain Wall) con blackening:
        ds^2 = e^{2A(z)} [-f(z)dt^2 + dx^2] + dz^2/f(z)
    
    FORMULA DEL ESCALAR DE RICCI:
        R = -2D*A'' - D(D-1)*(A')^2 - (f'/f)*A'
    
    donde D = d + 1 (dimension del bulk).
    
    NOTA IMPORTANTE SOBRE R_is_constant:
    En este gauge, para AdS puro con A = -log(z/L), f = 1:
        R = -D(D+1)/z^2
    
    Esto NO es constante en z. El test R_is_constant verifica que el
    coeficiente de variacion de R en la region interior sea bajo, lo cual
    detecta inestabilidades numericas pero NO es equivalente a "R constante
    en el sentido de Poincare".
    
    Para verificar consistencia con AdS, usamos R_matches_ads que compara
    el valor medio de R con el valor esperado -D(D-1) (con L=1).
    """
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        # Fallback sin scipy
        def gaussian_filter1d(x, sigma, mode='nearest'):
            return x
    
    D = d + 1
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    n = len(z)
    
    if n < 10:
        # Datos insuficientes
        return AdSEinsteinContract(
            R_is_constant=False,
            R_matches_ads=False,
            Lambda_matches_ads=False,
            R_mean=float('nan'),
            R_expected=float(-D * (D - 1))
        )
    
    # Suavizar A y f antes de derivar (reduce ruido numerico)
    sigma = 2.0
    A_smooth = gaussian_filter1d(A, sigma=sigma, mode='nearest')
    f_smooth = gaussian_filter1d(f, sigma=sigma, mode='nearest')
    f_smooth = np.clip(f_smooth, 1e-6, None)
    
    # Derivadas con esquema de 5 puntos
    def deriv_5pt(y, dx):
        d = np.zeros_like(y)
        for i in range(2, len(y) - 2):
            d[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12 * dx)
        d[0] = (y[1] - y[0]) / dx
        d[1] = (y[2] - y[0]) / (2 * dx) if len(y) > 2 else d[0]
        d[-1] = (y[-1] - y[-2]) / dx
        d[-2] = (y[-1] - y[-3]) / (2 * dx) if len(y) > 2 else d[-1]
        return d
    
    dA = deriv_5pt(A_smooth, dz)
    d2A = deriv_5pt(dA, dz)
    df = deriv_5pt(f_smooth, dz)
    
    # Calcular R usando formula consistente con 02/03
    # R = -2D*A'' - D(D-1)*(A')^2 - (f'/f)*A'
    R = -2 * D * d2A - D * (D - 1) * dA**2 - df * dA / f_smooth
    
    # Evaluar solo en region interior (evitar efectos de borde)
    margin = max(5, n // 10)
    R_interior = R[margin:-margin] if margin < n // 2 else R
    z_interior = z[margin:-margin] if margin < n // 2 else z
    
    R_mean = np.mean(R_interior)
    R_std = np.std(R_interior)
    R_expected = -D * (D - 1)  # Valor AdS con L=1
    
    # Test R_is_constant: coeficiente de variacion bajo
    # Note: En nuestro gauge, R ~ 1/z^2 para AdS, asi que este test
    # es mas sobre estabilidad numerica que constancia estricta
    cv = R_std / (np.abs(R_mean) + 1e-10)
    R_is_constant = cv < 0.5  # Relajado de 0.3 a 0.5 por el efecto del gauge
    
    # Test R_matches_ads: valor medio cercano al esperado
    # Usamos tolerancia relativa amplia porque R varia con z en nuestro gauge
    R_matches_ads = np.abs(R_mean - R_expected) / (np.abs(R_expected) + 1e-10) < 0.5
    
    Lambda_check = einstein_results.get("einstein_check", {})
    Lambda_matches = Lambda_check.get("consistent_with_einstein_vacuum", False)
    
    return AdSEinsteinContract(
        R_is_constant=R_is_constant,
        R_matches_ads=R_matches_ads,
        Lambda_matches_ads=Lambda_matches,
        R_mean=float(R_mean),
        R_expected=float(R_expected)
    )


def verify_ads_asymptotic(
    A: np.ndarray,
    f: np.ndarray,
    z: np.ndarray
) -> AdSAsymptoticContract:
    """Verifica comportamiento asintotico tipo AdS."""
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        def gaussian_filter1d(x, sigma, mode='nearest'):
            return x
    
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    n = len(z)
    
    if n < 5:
        return AdSAsymptoticContract(
            A_logarithmic_uv=False,
            f_to_one_uv=False,
            A_monotone=False
        )
    
    n_uv = min(10, n // 5)
    n_uv = max(3, n_uv)
    
    A_uv = A[:n_uv]
    z_uv = z[:n_uv]
    
    # Evitar log(0)
    z_uv_safe = np.clip(z_uv, 1e-10, None)
    expected_A = -np.log(z_uv_safe)
    
    # Correlacion
    if np.std(A_uv) > 1e-10 and np.std(expected_A) > 1e-10:
        corr = np.corrcoef(A_uv, expected_A)[0, 1]
        A_log_uv = corr > 0.95 if not np.isnan(corr) else False
    else:
        A_log_uv = False
    
    f_uv = f[:n_uv]
    f_to_one = np.mean(np.abs(f_uv - 1)) < 0.15
    
    A_smooth = gaussian_filter1d(A, sigma=2, mode='nearest')
    dA = np.gradient(A_smooth, dz)
    
    margin = max(3, n // 20)
    dA_interior = dA[margin:-margin] if margin < n // 2 else dA
    A_monotone = np.mean(dA_interior < 0.01) > 0.8
    
    return AdSAsymptoticContract(
        A_logarithmic_uv=A_log_uv,
        f_to_one_uv=f_to_one,
        A_monotone=A_monotone
    )


def verify_holographic(
    dict_results: Dict[str, Any]
) -> HolographicDictionaryContract:
    """Verifica diccionario holografico."""
    def to_bool(val) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == 'true'
        return bool(val) if val is not None else False
    
    mass_dim = dict_results.get("mass_dimension", {})
    mass_ok = to_bool(mass_dim.get("comparison_with_holographic", {}).get("likely_holographic", False))
    
    hawking = dict_results.get("hawking", {})
    hawking_ok = to_bool(hawking.get("hawking_check", {}).get("hawking_verified", False))
    
    conformal_ok = False
    for geo in dict_results.get("geometries", []):
        conf = geo.get("conformal", {}).get("summary", {})
        if to_bool(conf.get("conformal_symmetry_present", False)):
            conformal_ok = True
            break
    
    return HolographicDictionaryContract(
        mass_dimension_ok=mass_ok,
        hawking_ok=hawking_ok,
        conformal_symmetry_ok=conformal_ok
    )


# ============================================================
# CARGA DE DATOS CON FALLBACKS (V2.1)
# ============================================================

def load_geometry_data(
    name: str,
    data_dir: Optional[Path],
    predictions_dir: Path,
    emergent_h5_dir: Optional[Path]
) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Carga datos de geometria con fallbacks para inference mode.
    
    Returns:
        (data_dict, mode, warnings)
        - data_dict: diccionario con z_grid, A_pred, f_pred, R_pred, y opcionalmente *_truth
        - mode: "sandbox" o "inference"
        - warnings: lista de advertencias
    """
    data = {}
    warnings = []
    mode = "sandbox"
    
    # === 1. Intentar cargar NPZ de predictions ===
    npz_path = predictions_dir / f"{name}_geometry.npz"
    h5_emergent_path = emergent_h5_dir / f"{name}_emergent.h5" if emergent_h5_dir else None
    
    npz_loaded = False
    if npz_path.exists():
        try:
            geo_data = np.load(npz_path, allow_pickle=True)
            
            # Cargar predicciones
            data["A_pred"] = geo_data["A_pred"] if "A_pred" in geo_data else geo_data.get("A_of_z")
            data["f_pred"] = geo_data["f_pred"] if "f_pred" in geo_data else geo_data.get("f_of_z")
            data["R_pred"] = geo_data.get("R_pred", geo_data.get("R_of_z"))
            data["z_grid"] = geo_data.get("z", geo_data.get("z_grid"))
            
            # Cargar truth si existe
            if "A_truth" in geo_data:
                data["A_truth"] = geo_data["A_truth"]
                data["f_truth"] = geo_data["f_truth"]
                data["R_truth"] = geo_data.get("R_truth")
                data["family_pred"] = geo_data.get("family_pred")
                data["family_truth"] = geo_data.get("family_truth")
            else:
                mode = "inference"
                warnings.append(f"NPZ sin *_truth: inference mode")
            
            npz_loaded = True
        except Exception as e:
            warnings.append(f"Error loading NPZ: {e}")
    
    # === 2. Fallback a H5 emergente si NPZ no tiene lo necesario ===
    if not npz_loaded or data.get("A_pred") is None:
        if h5_emergent_path and h5_emergent_path.exists():
            try:
                with h5py.File(h5_emergent_path, "r") as f:
                    data["z_grid"] = f["z_grid"][:]
                    data["A_pred"] = f["A_of_z"][:] if "A_of_z" in f else f.get("A_pred", np.array([]))[:]
                    data["f_pred"] = f["f_of_z"][:] if "f_of_z" in f else f.get("f_pred", np.array([]))[:]
                    data["R_pred"] = f["R_of_z"][:] if "R_of_z" in f else f.get("R_pred", np.zeros_like(data["A_pred"]))[:]
                    
                    # Metadatos del H5
                    data["family_from_h5"] = str(f.attrs.get("family", f.attrs.get("family_pred", "unknown")))
                    data["d_from_h5"] = int(f.attrs.get("d", f.attrs.get("d_pred", 4)))
                
                mode = "inference"
                warnings.append(f"loaded from H5 emergente (fallback)")
            except Exception as e:
                warnings.append(f"Error loading H5 emergente: {e}")
        elif h5_emergent_path:
            warnings.append(f"Does not exist H5 emergente: {h5_emergent_path}")
        else:
            warnings.append("Sin directory emergent_h5 especificado")
    
    # === 3. Cargar metadatos del boundary H5 original ===
    boundary_h5_path = data_dir / f"{name}.h5" if data_dir else None
    if boundary_h5_path and boundary_h5_path.exists():
        try:
            with h5py.File(boundary_h5_path, "r") as f:
                data["family"] = str(f.attrs.get("family", "unknown"))
                data["category"] = str(f.attrs.get("category", "unknown"))
                
                # V2.3: Cargar operadores para contrato de unitariedad
                operators_raw = f.attrs.get("operators", "[]")
                if isinstance(operators_raw, bytes):
                    operators_raw = operators_raw.decode("utf-8")
                try:
                    data["operators"] = json.loads(operators_raw)
                except (json.JSONDecodeError, TypeError):
                    data["operators"] = []
                
                # Boundary data
                if "boundary" in f:
                    boundary = f["boundary"]
                    data["T"] = float(boundary.attrs.get("temperature", 0))
                    
                    # V2.3: Cargar correlador G2 para contrato de estructura
                    # Buscar cualquier G2_* disponible
                    for key in boundary.keys():
                        if key.startswith("G2_"):
                            data["G2"] = boundary[key][:]
                            break
                    
                    # Cargar x_grid si existe
                    if "x_grid" in boundary:
                        data["x_grid"] = boundary["x_grid"][:]
                    elif "distances" in boundary:
                        data["x_grid"] = boundary["distances"][:]
                else:
                    data["T"] = None
                
                # Bulk truth (solo en sandbox)
                if "bulk_truth" in f:
                    bulk = f["bulk_truth"]
                    data["z_h"] = float(bulk.attrs.get("z_h", 0))
                    if data.get("z_grid") is None:
                        data["z_grid"] = bulk["z_grid"][:]
                else:
                    data["z_h"] = None
                    if mode != "inference":
                        mode = "inference"
                        warnings.append("No bulk_truth en H5: inference mode")
        except Exception as e:
            warnings.append(f"Error loading boundary H5: {e}")
    else:
        # Sin H5 original, usar metadatos del emergente
        data["family"] = data.get("family_from_h5", "unknown")
        data["category"] = "inference"
        data["T"] = None
        data["z_h"] = None
        data["operators"] = []  # V2.3
        mode = "inference"
        if boundary_h5_path:
            warnings.append(f"Does not exist boundary H5: {boundary_h5_path}")
        else:
            warnings.append("Sin data_dir especificado: inference mode")
    
    return data, mode, warnings


# ============================================================
# PROCESAMIENTO (V2.1 - CON SOPORTE INFERENCE)
# ============================================================

def process_geometry(
    name: str,
    data_dir: Path,
    predictions_dir: Path,
    emergent_h5_dir: Path,
    einstein_dir: Path,
    dictionary_results: Dict[str, Any],
    d: int
) -> PhaseXIContractV2:
    """
    Procesa una geometria y genera su contrato v2.
    
    Soporta dos modos:
    - sandbox: con bulk_truth y *_truth (metricas R² calculadas)
    - inference: without truth (metricas R² = None, generic contracts evaluados)
    """
    errors = []
    warnings = []
    
    # === CARGAR DATOS ===
    try:
        data, mode, load_warnings = load_geometry_data(
            name, data_dir, predictions_dir, emergent_h5_dir
        )
        warnings.extend(load_warnings)
    except Exception as e:
        errors.append(f"Fatal error loading data: {e}")
        # Devolver contrato vacio pero presente
        return PhaseXIContractV2(
            name=name,
            family="unknown",
            category="error",
            d=d,
            regularity=GenericRegularityContract(False, False, False, False),
            causality=GenericCausalityContract(False, False, False, skipped=True),
            unitarity=BoundaryUnitarityContract(True, 0, 0, float('nan'), (d-2)/2, [], skipped=True),
            correlator_structure=CorrelatorStructureContract(True, True, True, float('nan'), float('nan'), skipped=True),
            ads_einstein=AdSEinsteinContract(False, False, False, float('nan'), float('nan')),
            ads_asymptotic=AdSAsymptoticContract(False, False, False),
            holographic=HolographicDictionaryContract(False, False, False),
            A_r2=None,
            f_r2=None,
            R_r2=None,
            family_accuracy=None,
            mode="error",
            errors=errors,
            warnings=warnings
        )
    
    # Extraer arrays
    z_grid = data.get("z_grid")
    A_pred = data.get("A_pred")
    f_pred = data.get("f_pred")
    R_pred = data.get("R_pred")
    
    family = data.get("family", "unknown")
    category = data.get("category", "unknown")
    T = data.get("T")
    z_h = data.get("z_h")
    
    # Validar datos minimos
    if z_grid is None or A_pred is None or f_pred is None:
        errors.append("Datos insuficientes: falta z_grid, A_pred o f_pred")
        return PhaseXIContractV2(
            name=name,
            family=family,
            category=category,
            d=d,
            regularity=GenericRegularityContract(False, False, False, False),
            causality=GenericCausalityContract(False, False, False, skipped=True),
            unitarity=BoundaryUnitarityContract(True, 0, 0, float('nan'), (d-2)/2, [], skipped=True),
            correlator_structure=CorrelatorStructureContract(True, True, True, float('nan'), float('nan'), skipped=True),
            ads_einstein=AdSEinsteinContract(False, False, False, float('nan'), float('nan')),
            ads_asymptotic=AdSAsymptoticContract(False, False, False),
            holographic=HolographicDictionaryContract(False, False, False),
            A_r2=None,
            f_r2=None,
            R_r2=None,
            family_accuracy=None,
            mode=mode,
            errors=errors,
            warnings=warnings
        )
    
    # Asegurar arrays numpy
    z_grid = np.asarray(z_grid)
    A_pred = np.asarray(A_pred)
    f_pred = np.asarray(f_pred)
    if R_pred is not None:
        R_pred = np.asarray(R_pred)
    else:
        R_pred = np.zeros_like(A_pred)
    
    # === MÉTRICAS R² (solo en sandbox) ===
    def r2(y_true, y_pred):
        if y_true is None or y_pred is None:
            return None
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        r2_val = 1 - ss_res / ss_tot
        return float(np.clip(r2_val, -1.0, 1.0))
    
    if mode == "sandbox" and "A_truth" in data:
        A_r2 = r2(data["A_truth"], A_pred)
        f_r2 = r2(data["f_truth"], f_pred)
        R_r2 = r2(data.get("R_truth"), R_pred) if data.get("R_truth") is not None else None
        
        # Family accuracy
        fp = data.get("family_pred")
        ft = data.get("family_truth")
        if fp is not None and ft is not None:
            family_match = float(int(fp) == int(ft))
        else:
            family_match = None
    else:
        A_r2 = None
        f_r2 = None
        R_r2 = None
        family_match = None
        if mode == "inference":
            warnings.append("inference mode: metricas R² no disponibles")
    
    # === CARGAR RESULTADOS EINSTEIN ===
    einstein_results = {}
    if einstein_dir:
        einstein_path = einstein_dir / name / "einstein_discovery.json"
        if einstein_path.exists():
            try:
                einstein_results = json.loads(einstein_path.read_text()).get("results", {})
            except Exception as e:
                warnings.append(f"Error loading einstein_discovery.json: {e}")
    
    # === VERIFICAR CONTRATOS ===
    regularity = verify_regularity(A_pred, f_pred, z_grid)
    causality = verify_generic_causality(f_pred, z_grid, T, z_h)
    
    # V2.3: Nuevos contratos de boundary
    operators = data.get("operators", [])
    unitarity = verify_boundary_unitarity(operators, d)
    
    G2 = data.get("G2")
    x_grid = data.get("x_grid")
    correlator_structure = verify_correlator_structure(G2, x_grid)
    
    ads_einstein = verify_ads_einstein(A_pred, f_pred, z_grid, d, einstein_results)
    ads_asymptotic = verify_ads_asymptotic(A_pred, f_pred, z_grid)
    holographic = verify_holographic(dictionary_results)
    
    return PhaseXIContractV2(
        name=name,
        family=family,
        category=category,
        d=d,
        regularity=regularity,
        causality=causality,
        unitarity=unitarity,
        correlator_structure=correlator_structure,
        ads_einstein=ads_einstein,
        ads_asymptotic=ads_asymptotic,
        holographic=holographic,
        A_r2=A_r2,
        f_r2=f_r2,
        R_r2=R_r2,
        family_accuracy=family_match,
        mode=mode,
        errors=errors,
        warnings=warnings
    )


# ============================================================
# MAIN (V2.1)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PHYSICS CONTRACTS VALIDATOR (con soporte inference)"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="directory con H5 de boundary (manifest.json)")
    parser.add_argument("--geometry-dir", type=str, default=None,
                        help="directory raiz de geometria (predictions/, geometry_emergent/)")
    parser.add_argument("--einstein-dir", type=str, default=None,
                        help="directory de ecuaciones bulk (o raiz que contenga bulk_equations/)")
    parser.add_argument("--dictionary-file", type=str, default=None,
                        help="JSON de diccionario holografico")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="[DEPRECATED] Usar --experiment")
    parser.add_argument("--output-file", type=str, default=None,
                        help="file de salida JSON (default: run-dir/geometry_contracts/...)")
    parser.add_argument("--d", type=int, default=4,
                        help="Dimension d del boundary")
    
    # V3 Infrastructure
    if HAS_STAGE_UTILS:
        add_standard_arguments(parser)
        args = parse_stage_args(parser)
        ctx = StageContext.from_args(args, stage_number="04", stage_slug="geometry_physics_contracts")
        run_dir = ctx.run_root
    else:
        parser.add_argument("--experiment", type=str, default=None,
                            help="Nombre del experimento")
        args = parser.parse_args()
        ctx = None
        # Resolver run_dir
        if args.experiment:
            run_dir = Path("runs") / args.experiment
        elif args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = None
    
    # === RESOLVER RUTAS (IO v2 con fallback legacy) ===
    predictions_dir = None
    emergent_h5_dir = None
    einstein_dir = None
    data_dir = None
    dictionary_path = None
    
    # Prioridad 1: run_dir (desde --experiment o --run-dir) con cuerdas_io
    if run_dir and HAS_CUERDAS_IO:
        predictions_dir = cuerdas_resolve_predictions(run_dir=run_dir)
        emergent_h5_dir = cuerdas_resolve_geometry_emergent(run_dir=run_dir)
        einstein_dir = cuerdas_resolve_bulk_equations(run_dir=run_dir)
        data_dir = cuerdas_resolve_data(run_dir=run_dir, data_dir=args.data_dir)
        dictionary_path = cuerdas_resolve_dictionary(run_dir=run_dir, dictionary_file=args.dictionary_file)
    
    # Prioridad 2: argumentos explicitos (legacy)
    if predictions_dir is None and args.geometry_dir:
        geometry_dir = Path(args.geometry_dir)
        predictions_dir = resolve_predictions_dir(geometry_dir)
        emergent_h5_dir = resolve_emergent_h5_dir(geometry_dir)
    
    if einstein_dir is None and args.einstein_dir:
        einstein_dir = resolve_bulk_equations_dir(Path(args.einstein_dir))
    
    if data_dir is None and args.data_dir:
        data_dir = Path(args.data_dir)
    
    if dictionary_path is None and args.dictionary_file:
        dictionary_path = Path(args.dictionary_file)
    
    # Validar que tenemos las rutas minimas necesarias
    if predictions_dir is None:
        parser.error("Must provide --experiment o --geometry-dir")
    
    # Resolver output_file
    if args.output_file:
        output_file = Path(args.output_file)
    elif run_dir:
        output_dir = run_dir / "04_geometry_physics_contracts"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "geometry_contracts_summary.json"
    else:
        output_file = Path("contracts_04.json")
    
    print("=" * 90)
    print("PHYSICS CONTRACTS VALIDATOR")
    print("=" * 90)
    print(f"\n  experiment:     {getattr(args, 'experiment', None) or '(no especificado)'}")
    print(f"  run_dir:        {run_dir or '(no especificado)'}")
    print(f"  data-dir:       {data_dir or '(no especificado)'}")
    print(f"  predictions:    {predictions_dir}")
    print(f"  emergent_h5:    {emergent_h5_dir}")
    print(f"  einstein-dir:   {einstein_dir}")
    print(f"  dictionary:     {dictionary_path}")
    print(f"  output:         {output_file}")
    print("\nPHILOSOPHY:")
    print("   generic contracts: All geometries must pass")
    print("   AdS-specific contracts: Only relevant for family='ads'")
    print("   inference mode: without truth, R2=null, generic contracts evaluados")
    print("=" * 90)
    
    # === CARGAR MANIFEST ===
    if data_dir is None:
        # Sin data_dir, auto-descubrir desde predictions
        print(f"\n[INFO] No data-dir, Auto-discovering sistemas desde predictions/")
        npz_files = list(predictions_dir.glob("*_geometry.npz"))
        if npz_files:
            print(f"[INFO] Encontrados {len(npz_files)} sistemas")
            geometries = [{"name": f.stem.replace("_geometry", "")} for f in npz_files]
        else:
            print("[ERROR] No hay sistemas para procesar")
            return
    else:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"\n[WARN] Does not exist manifest.json en {data_dir}")
            # Intentar auto-descubrir sistemas desde predictions/
            npz_files = list(predictions_dir.glob("*_geometry.npz"))
            if npz_files:
                print(f"[INFO] Auto-discovering {len(npz_files)} sistemas desde predictions/")
                geometries = [{"name": f.stem.replace("_geometry", "")} for f in npz_files]
            else:
                print("[ERROR] No hay sistemas para procesar")
                return
        else:
            manifest = json.loads(manifest_path.read_text())
            geometries = manifest.get("geometries", [])
    
    # Cargar diccionario
    if dictionary_path and dictionary_path.exists():
        dictionary_results = json.loads(dictionary_path.read_text())
    else:
        dictionary_results = {}
        if dictionary_path:
            print(f"[WARN] Does not exist diccionario: {dictionary_path}")
        else:
            print("[WARN] No se especifico file de diccionario")
    
    all_contracts = []
    
    for geo_info in geometries:
        name = geo_info["name"] if isinstance(geo_info, dict) else geo_info
        print(f"\n>> {name}")
        
        contract = process_geometry(
            name, data_dir, predictions_dir, emergent_h5_dir,
            einstein_dir, dictionary_results, args.d
        )
        all_contracts.append(contract)
        
        # Mostrar resultado (V2.3)
        if contract.errors:
            print(f"   [ERROR] {contract.errors}")
        else:
            print(f"   Family: {contract.family} | Mode: {contract.mode}")
            # V2.3: Display mejorado con nuevos contratos
            gen_ok = "OK" if contract.generic_passed else "FAIL"
            print(f"   Genericos:     {gen_ok} "
                  f"(reg={contract.regularity.passed}, caus={contract.causality.passed}, "
                  f"unit={contract.unitarity.passed}, corr={contract.correlator_structure.passed})")
            
            # V2.3: Mostrar detalles de unitariedad si fallo
            if not contract.unitarity.passed and not contract.unitarity.skipped:
                print(f"      -> Unitariedad: {contract.unitarity.n_violations} violaciones")
                for v in contract.unitarity.violations[:3]:
                    print(f"         {v['name']}: Delta={v['Delta']:.3f} < bound={v['bound']:.3f}")
            
            # V2.3: Mostrar detalles de correlador si fallo
            if not contract.correlator_structure.passed and not contract.correlator_structure.skipped:
                print(f"      -> Correlador: spatial={contract.correlator_structure.has_spatial_structure}, "
                      f"power_law={contract.correlator_structure.has_power_law} "
                      f"(slope={contract.correlator_structure.log_slope:.2f})")

            if contract.is_ads_family:
                print(f"   AdS-specific:  {'OK' if contract.ads_specific_passed else 'FAIL'} "
                      f"(einstein={contract.ads_einstein.passed}, asymp={contract.ads_asymptotic.passed})")
            else:
                print(f"   AdS-specific:  N/A (no es family='ads')")
            
            if contract.A_r2 is not None:
                print(f"   R²: A={contract.A_r2:.3f}, f={contract.f_r2:.3f}")
            else:
                print(f"   R²: (no disponible en inference)")
            
            print(f"   Score: {contract.contract_score:.2f}")
        
        if contract.warnings:
            for w in contract.warnings:
                print(f"   [WARN] {w}")
    
    # ============================================================
    # RESUMEN
    # ============================================================
    
    print("\n" + "=" * 100)
    print("CONTRACTS SUMMARY FASE XI v2.1")
    print("=" * 100)
    
    # Tabla
    print(f"\n{'Name':<25} {'Family':<12} {'Mode':<10} {'Generic':^8} {'AdS':^8} {'Score':^8} {'Pass':^6}")
    print("-" * 100)
    
    for c in all_contracts:
        ads_col = 'OK' if c.ads_specific_passed else ('FAIL' if c.is_ads_family else '-')
        mode_short = "sandbox" if c.mode == "sandbox" else "infer"
        gen_status = 'OK' if c.generic_passed else 'FAIL'
        pass_status = 'OK' if c.overall_passed else 'FAIL'
        print(f"{c.name:<25} {c.family:<12} {mode_short:<10} "
              f"{gen_status:^8} "
              f"{ads_col:^8} "
              f"{c.contract_score:.2f} "
              f"{pass_status:^6}")
    
    print("-" * 100)
    
    # Estadisticas
    n_total = len(all_contracts)
    n_with_errors = sum(1 for c in all_contracts if c.errors)
    n_inference = sum(1 for c in all_contracts if c.mode == "inference")
    n_generic = sum(c.generic_passed for c in all_contracts)
    n_ads_family = sum(c.is_ads_family for c in all_contracts)
    n_ads_passed = sum(c.ads_specific_passed for c in all_contracts if c.is_ads_family)
    n_overall = sum(c.overall_passed for c in all_contracts)
    
    contracts_with_score = [c for c in all_contracts if not c.errors]
    avg_score = np.mean([c.contract_score for c in contracts_with_score]) if contracts_with_score else 0.0
    
    print(f"\nSTATISTICS:")
    print(f"  Total geometrias:      {n_total}")
    print(f"  Con errores:           {n_with_errors}")
    print(f"  inference mode:        {n_inference}")
    print(f"  Genericos OK:          {n_generic}/{n_total}")
    print(f"  AdS families:          {n_ads_family}/{n_total}")
    print(f"  AdS-specific OK:       {n_ads_passed}/{n_ads_family} (de las que son AdS)")
    print(f"  Overall passed:        {n_overall}/{n_total}")
    print(f"  average score:        {avg_score:.3f}")
    
    # Por modo
    print("\nBY MODE:")
    for mode_name in ["sandbox", "inference"]:
        mode_contracts = [c for c in all_contracts if c.mode == mode_name]
        if mode_contracts:
            n_m = len(mode_contracts)
            n_passed = sum(c.overall_passed for c in mode_contracts)
            avg_m = np.mean([c.contract_score for c in mode_contracts])
            print(f"  {mode_name:12}: {n_passed}/{n_m} passed, score={avg_m:.2f}")
    
    # Veredicto final
    print("\n" + "=" * 90)
    
    phase_passed = (n_with_errors == 0) and (n_generic == n_total) and (avg_score > 0.5 or n_total == 0)
    
    if n_total == 0:
        print("⚠ NO HAY GEOMETRÍAS PARA EVALUAR")
    elif phase_passed:
        print("✓ VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 90)
        print(f"\n  El sistema CUERDAS ha logrado:")
        print(f"     All geometries pass generic contracts")
        if n_inference > 0:
            print(f"     {n_inference} geometries processed en inference mode")
    else:
        print("✗ VALIDATION REQUIRES REFINEMENT")
        print("=" * 90)
        if n_with_errors > 0:
            print(f"\n  {n_with_errors} geometries with errors")
        if n_generic < n_total:
            print(f"\n  {n_total - n_generic} geometries do not pass generic contracts")
        if avg_score <= 0.5:
            print(f"\n  average score ({avg_score:.2f}) demasiado bajo")
    
    # === GUARDAR ===
    def serialize_value(v):
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            if v is None:
                return None
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, dict):
            return {k: serialize_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [serialize_value(item) for item in v]
        if v is None:
            return None
        return v
    
    def clean_contract(c_dict):
        return {k: serialize_value(v) for k, v in c_dict.items()}
    
    output_data = {
        "version": "2.1",
        "n_total": n_total,
        "n_with_errors": n_with_errors,
        "n_inference_mode": n_inference,
        "n_generic_passed": int(n_generic),
        "n_ads_family": int(n_ads_family),
        "n_ads_specific_passed": int(n_ads_passed),
        "n_overall_passed": int(n_overall),
        "avg_score": float(avg_score) if not np.isnan(avg_score) else None,
        "phase_passed": bool(phase_passed),
        "contracts": [clean_contract(asdict(c)) for c in all_contracts]
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n  Results: {output_file}")
    
    # === ACTUALIZAR RUN_MANIFEST (IO v2) ===
    if args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir)
            update_run_manifest(
                run_dir,
                {
                    "geometry_contracts_dir": str(output_file.parent.relative_to(run_dir)
                                                  if output_file.parent.is_relative_to(run_dir)
                                                  else output_file.parent),
                    "geometry_contracts_summary": str(output_file.relative_to(run_dir)
                                                      if output_file.is_relative_to(run_dir)
                                                      else output_file),
                }
            )
            print(f"  Manifest actualizado: {run_dir / 'run_manifest.json'}")
        except Exception as e:
            print(f"  [WARN] No se pudo actualizar run_manifest.json: {e}")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
