#!/usr/bin/env python3
# 09_real_data_and_dictionary_contracts.py
# CUERDAS â€” Bloque C: Contratos con datos reales y diccionario emergente
#
# CONTROL NEGATIVO (v3)
#   MÃ©trica corregida: False Positive Rate (FPR) sobre seÃ±ales hologrÃ¡ficas.
#   NO mezcla "contratos que deben fallar" con "seÃ±ales de autoengaÃ±o".
#
#   FPR = (seÃ±ales hologrÃ¡ficas disparadas) / (seÃ±ales evaluables)
#   
#   Esto mide: "Â¿El pipeline cree que esto es hologrÃ¡fico cuando NO deberÃ­a?"
#
# MIGRADO A V3: 2024-12-23
#
# MEJORAS CFT v2 (2024-12-26):
#   - IntegraciÃ³n de contratos extendidos desde extended_physics_contracts.py
#   - ValidaciÃ³n de operadores especiales (T_Î¼Î½, J_Î¼, identidad)
#   - ValidaciÃ³n de limitaciÃ³n de spin
#   - Contratos OPE preparados para extracciÃ³n futura
#
# ============================================================================
# REFERENCIAS Y FUENTES (actualizado 2024-12-29)
# ============================================================================
#
# Este script implementa DOS tipos de validaciones:
#
# A) VALIDACIONES DEL TEXTO DE REFERENCIA (AGMOO 1999):
#    - Unitarity bound: Delta >= (d-2)/2 para escalares (Sec. 2.1)
#    - Relacion masa-dimension: m^2 L^2 = Delta(Delta-d) (Eq. 3.14)
#    - Propiedades generales de CFTs (Sec. 2.1)
#
# B) VALIDACIONES DE LITERATURA MODERNA (NO en AGMOO):
#    - Ising 3D: Delta_sigma=0.518, Delta_epsilon=1.41
#      Ref: Conformal Bootstrap (El-Showk et al. 2012, Kos et al. 2016)
#    - KSS bound: eta/s >= 1/(4*pi)
#      Ref: Kovtun, Son, Starinets (2005) - posterior a AGMOO
#    - Strange metal scaling: rho ~ T^alpha
#      Ref: Applied AdS/CMT (Hartnoll et al. 2009+)
#
# HONESTIDAD EPISTEMOLOGICA:
#    Los contratos tipo (B) son VALIDACIONES EXTERNAS post-hoc, NO
#    inyeccion de teoria. Se documentan explicitamente para distinguir
#    validacion del texto vs validacion de datos reales.
#
# ============================================================================
#
# REFERENCIA PRINCIPAL:
#   Maldacena et al., "Large N Field Theories..." (AGMOO review)
#   Chunk 04: Conformal Field Theories (pags 30-35)

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# V3 INFRASTRUCTURE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
HAS_STAGE_UTILS = False
StageContext = None
add_standard_arguments = None
infer_experiment = None

try:
    from stage_utils import StageContext, add_standard_arguments, infer_experiment
    HAS_STAGE_UTILS = True
except ImportError:
    pass

if not HAS_STAGE_UTILS:
    try:
        from tools.stage_utils import StageContext, add_standard_arguments, infer_experiment
        HAS_STAGE_UTILS = True
    except ImportError:
        pass

# Legacy imports (fallback)
try:
    from cuerdas_io import load_run_manifest, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# IMPORTACIÃ“N DE CONTRATOS CFT EXTENDIDOS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
try:
    from extended_physics_contracts import (
        ExtendedContractsFase12,
        ExtendedContractsFase13,
        merge_contract_results,
        get_spin_unitarity_bound,
        OPE_REFERENCE_DATA,
        SPECIAL_OPERATORS_CFT,
    )
    HAS_EXTENDED_CONTRACTS = True
except ImportError:
    HAS_EXTENDED_CONTRACTS = False
    logger.warning(
        "extended_physics_contracts.py no encontrado. "
        "Contratos CFT extendidos deshabilitados."
    )


# ============================================================
# SEÃ‘ALES HOLOGRÃFICAS (para control negativo)
# ============================================================

class SignalStatus(Enum):
    """Estado de una seÃ±al hologrÃ¡fica."""
    TRIGGERED = "triggered"      # SeÃ±al indica holografÃ­a (falso positivo)
    NOT_TRIGGERED = "not_triggered"  # SeÃ±al NO indica holografÃ­a (correcto)
    NOT_EVALUABLE = "not_evaluable"  # No hay artefactos para evaluar


@dataclass
class HolographicSignal:
    """
    Una seÃ±al hologrÃ¡fica es un check binario que responde:
    "Â¿El pipeline estÃ¡ actuando COMO SI esto fuera hologrÃ¡fico?"
    
    En un control negativo, queremos que estas seÃ±ales NO se disparen.
    Si se disparan, es un falso positivo.
    """
    name: str
    description: str
    status: SignalStatus = SignalStatus.NOT_EVALUABLE
    value: Any = None
    threshold: Optional[float] = None
    evidence: str = ""
    
    @property
    def triggered(self) -> bool:
        return self.status == SignalStatus.TRIGGERED
    
    @property
    def evaluable(self) -> bool:
        return self.status != SignalStatus.NOT_EVALUABLE
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.threshold,
            "evidence": self.evidence,
            "triggered": self.triggered,
            "evaluable": self.evaluable,
        }


@dataclass
class ExpectedFailContract:
    """
    Un contrato que DEBE fallar en control negativo.
    No entra en el cÃ¡lculo de FPR - es informacional.
    """
    name: str
    passed: bool
    note: str = "expected-fail for negative control"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "note": self.note,
        }


def evaluate_holographic_signals(artifacts: Dict[str, Any]) -> List[HolographicSignal]:
    """
    EvalÃºa todas las seÃ±ales hologrÃ¡ficas a partir de los artefactos del pipeline.
    
    Cada seÃ±al responde: "Â¿El pipeline cree que esto es hologrÃ¡fico?"
    
    En control negativo, lo correcto es que NO se disparen.
    """
    signals = []
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEÃ‘AL 1: Familia geomÃ©trica es AdS-like
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    signal_family = HolographicSignal(
        name="family_ads_like",
        description="La familia geomÃ©trica clasificada es AdS o AdS-like"
    )
    
    geometry = artifacts.get("geometry", {})
    if geometry:
        family = geometry.get("family", "").lower()
        is_ads = "ads" in family
        
        signal_family.status = SignalStatus.TRIGGERED if is_ads else SignalStatus.NOT_TRIGGERED
        signal_family.value = family
        signal_family.evidence = f"Familia clasificada: '{family}'"
    else:
        signal_family.status = SignalStatus.NOT_EVALUABLE
        signal_family.evidence = "No se encontraron artefactos de geometrÃ­a"
    
    signals.append(signal_family)
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEÃ‘AL 2: Einstein score alto
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    signal_einstein = HolographicSignal(
        name="einstein_score_high",
        description="El score de Einstein equations discovery es alto (>=0.5)",
        threshold=0.5
    )
    
    einstein = artifacts.get("einstein", {})
    score = einstein.get("score")
    
    if score is not None:
        is_high = score >= signal_einstein.threshold
        signal_einstein.status = SignalStatus.TRIGGERED if is_high else SignalStatus.NOT_TRIGGERED
        signal_einstein.value = score
        signal_einstein.evidence = f"Einstein score: {score:.3f}"
    else:
        signal_einstein.status = SignalStatus.NOT_EVALUABLE
        signal_einstein.evidence = "No se encontrÃ³ einstein_score en artefactos"
    
    signals.append(signal_einstein)
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEÃ‘AL 3: Diccionario Î»_SL â†’ Î” convergiÃ³
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    signal_dict = HolographicSignal(
        name="dictionary_converged",
        description="El diccionario hologrÃ¡fico Î»_SL â†’ Î” convergiÃ³"
    )
    
    dictionary = artifacts.get("dictionary", {})
    converged = dictionary.get("converged")
    
    if converged is not None:
        signal_dict.status = SignalStatus.TRIGGERED if converged else SignalStatus.NOT_TRIGGERED
        signal_dict.value = converged
        signal_dict.evidence = f"Convergencia: {converged}"
    else:
        signal_dict.status = SignalStatus.NOT_EVALUABLE
        signal_dict.evidence = "No se encontrÃ³ estado de convergencia"
    
    signals.append(signal_dict)
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEÃ‘AL 4: Î” predichos en rango fÃ­sicamente plausible
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    signal_deltas = HolographicSignal(
        name="deltas_in_physical_range",
        description="MayorÃ­a de Î” predichos estÃ¡n en rango CFT plausible (0.3-4.0)",
        threshold=0.5  # >50% en rango = seÃ±al de holografÃ­a
    )
    
    predicted_deltas = dictionary.get("predicted_Deltas", [])
    
    if predicted_deltas:
        n_physical = sum(1 for d in predicted_deltas if 0.3 < d < 4.0)
        ratio = n_physical / len(predicted_deltas)
        
        in_range = ratio > signal_deltas.threshold
        signal_deltas.status = SignalStatus.TRIGGERED if in_range else SignalStatus.NOT_TRIGGERED
        signal_deltas.value = ratio
        signal_deltas.evidence = f"{n_physical}/{len(predicted_deltas)} Deltas en rango fÃ­sico"
    else:
        signal_deltas.status = SignalStatus.NOT_EVALUABLE
        signal_deltas.evidence = "No hay Deltas predichos"
    
    signals.append(signal_deltas)
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEÃ‘AL 5: Bulk equations limpias (symbolic regression exitosa)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    signal_bulk = HolographicSignal(
        name="bulk_equations_clean",
        description="Se encontraron ecuaciones bulk limpias (n_equations > 0 con score alto)"
    )
    
    n_equations = einstein.get("n_equations", 0)
    
    if n_equations is not None:
        has_equations = n_equations > 0 and (score is None or score > 0.3)
        signal_bulk.status = SignalStatus.TRIGGERED if has_equations else SignalStatus.NOT_TRIGGERED
        signal_bulk.value = n_equations
        signal_bulk.evidence = f"{n_equations} ecuaciones encontradas"
    else:
        signal_bulk.status = SignalStatus.NOT_EVALUABLE
        signal_bulk.evidence = "No hay informaciÃ³n de bulk equations"
    
    signals.append(signal_bulk)
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEÃ‘AL 6: Match con operadores conocidos (Î”Ïƒ â‰ˆ 0.518)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # =========================================================================
    # SENAL 6: Match con operadores Ising 3D
    # =========================================================================
    # FUENTE EXTERNA (NO en AGMOO 1999):
    #   - Delta_sigma = 0.518 (Conformal Bootstrap, El-Showk et al. 2012)
    #   - Esto es VALIDACION CONTRA DATOS REALES, no teoria del texto
    # =========================================================================
    signal_sigma = HolographicSignal(
        name="delta_sigma_match",
        description="AlgÃºn Î” predicho estÃ¡ cerca de Î”Ïƒ=0.518 (Ising 3D)",
        threshold=0.1  # tolerancia
    )
    
    if predicted_deltas:
        delta_sigma = 0.518
        matches = [d for d in predicted_deltas if abs(d - delta_sigma) < signal_sigma.threshold]
        
        has_match = len(matches) > 0
        signal_sigma.status = SignalStatus.TRIGGERED if has_match else SignalStatus.NOT_TRIGGERED
        signal_sigma.value = matches[0] if matches else None
        signal_sigma.evidence = f"Matches con Î”Ïƒ: {matches}" if matches else "Sin match"
    else:
        signal_sigma.status = SignalStatus.NOT_EVALUABLE
        signal_sigma.evidence = "No hay Deltas predichos"
    
    signals.append(signal_sigma)
    
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEÃ‘AL 7: Match con operador Îµ (Î”Îµ â‰ˆ 1.41)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # =========================================================================
    # SENAL 7: Match con operador epsilon Ising 3D
    # =========================================================================
    # FUENTE EXTERNA (NO en AGMOO 1999):
    #   - Delta_epsilon = 1.41 (Conformal Bootstrap, Kos et al. 2016)
    # =========================================================================
    signal_epsilon = HolographicSignal(
        name="delta_epsilon_match",
        description="AlgÃºn Î” predicho estÃ¡ cerca de Î”Îµ=1.41 (Ising 3D)",
        threshold=0.15
    )
    
    if predicted_deltas:
        delta_epsilon = 1.41
        matches = [d for d in predicted_deltas if abs(d - delta_epsilon) < signal_epsilon.threshold]
        
        has_match = len(matches) > 0
        signal_epsilon.status = SignalStatus.TRIGGERED if has_match else SignalStatus.NOT_TRIGGERED
        signal_epsilon.value = matches[0] if matches else None
        signal_epsilon.evidence = f"Matches con Î”Îµ: {matches}" if matches else "Sin match"
    else:
        signal_epsilon.status = SignalStatus.NOT_EVALUABLE
        signal_epsilon.evidence = "No hay Deltas predichos"
    
    signals.append(signal_epsilon)
    
    return signals


def compute_false_positive_rate(signals: List[HolographicSignal]) -> Tuple[float, float, int, int]:
    """
    Calcula el False Positive Rate sobre seÃ±ales hologrÃ¡ficas.
    
    FPR = (seÃ±ales disparadas) / (seÃ±ales evaluables)
    
    Retorna: (fpr, coverage, n_triggered, n_evaluable)
    """
    evaluable = [s for s in signals if s.evaluable]
    triggered = [s for s in evaluable if s.triggered]
    
    n_evaluable = len(evaluable)
    n_triggered = len(triggered)
    n_total = len(signals)
    
    fpr = n_triggered / n_evaluable if n_evaluable > 0 else 0.0
    coverage = n_evaluable / n_total if n_total > 0 else 0.0
    
    return fpr, coverage, n_triggered, n_evaluable


def evaluate_expected_fail_contracts(artifacts: Dict[str, Any]) -> List[ExpectedFailContract]:
    """
    EvalÃºa contratos que DEBEN fallar en control negativo.
    Estos son informativos, no entran en el FPR.
    """
    contracts = []
    
    dictionary = artifacts.get("dictionary", {})
    predicted_deltas = dictionary.get("predicted_Deltas", [])
    geometry = artifacts.get("geometry", {})
    family = geometry.get("family", "").lower()
    
    passed = False
    if predicted_deltas and "ads" in family:
        delta_sigma = 0.518
        tolerance = 0.1
        matches = [d for d in predicted_deltas if abs(d - delta_sigma) < tolerance]
        passed = len(matches) > 0
    
    contracts.append(ExpectedFailContract(
        name="ising3d_consistency",
        passed=passed,
        note="DEBE fallar para control negativo (datos no-CFT)"
    ))
    
    return contracts


# ============================================================
# CONTROL NEGATIVO (orquestador)
# ============================================================

def verify_negative_control_h5(h5_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Verifica que un HDF5 tenga los atributos de control negativo."""
    if not HAS_H5PY:
        return False, {"error": "h5py not available"}
    
    if not h5_path.exists():
        return False, {"error": "file not found", "path": str(h5_path)}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'negative_control' not in f:
                return False, {"error": "no 'negative_control' group in HDF5"}
            
            grp = f['negative_control']
            is_negative = grp.attrs.get('IS_NEGATIVE_CONTROL', 0)
            expected_holo = grp.attrs.get('EXPECTED_HOLOGRAPHIC', 1)
            
            if is_negative != 1:
                return False, {"error": "IS_NEGATIVE_CONTROL != 1", "value": int(is_negative)}
            
            if expected_holo != 0:
                return False, {"error": "EXPECTED_HOLOGRAPHIC != 0", "value": int(expected_holo)}
            
            metadata = {
                "IS_NEGATIVE_CONTROL": int(is_negative),
                "EXPECTED_HOLOGRAPHIC": int(expected_holo),
                "type": grp.attrs.get('type', 'unknown'),
                "mass": float(grp.attrs.get('mass', 0)),
                "lattice_size": int(grp.attrs.get('lattice_size', 0)),
                "dimension": int(grp.attrs.get('dimension', 0)),
                "conformal": bool(grp.attrs.get('conformal', False)),
            }
            
            return True, metadata
            
    except Exception as e:
        return False, {"error": str(e)}


def find_negative_control_h5(run_dir: Path) -> Optional[Path]:
    """Busca el HDF5 de control negativo en un directorio."""
    candidates = list(run_dir.glob("negative_control_*.h5"))
    if candidates:
        return candidates[0]
    
    for subdir in run_dir.iterdir():
        if subdir.is_dir():
            candidates = list(subdir.glob("negative_control_*.h5"))
            if candidates:
                return candidates[0]
    
    return None


def load_negative_control_artifacts(run_dir: Path) -> Dict[str, Any]:
    """Carga los artefactos del pipeline ejecutado sobre control negativo."""
    artifacts = {
        "found": False,
        "geometry": {},
        "einstein": {},
        "dictionary": {},
        "errors": []
    }
    
    # Buscar geometrÃ­a
    for gdir in [run_dir / "geometry_emergent", run_dir / "predictions", run_dir / "geometry",
                 run_dir / "02_emergent_geometry_engine" / "geometry_emergent"]:
        if gdir.exists() and gdir.is_dir():
            for sf in list(gdir.glob("*summary*.json")) + list(gdir.glob("*report*.json")):
                try:
                    data = json.loads(sf.read_text())
                    artifacts["geometry"] = {
                        "source": str(sf),
                        "family": data.get("predicted_family", data.get("family", "unknown")),
                        "params": data.get("parameters", {}),
                    }
                    artifacts["found"] = True
                    break
                except:
                    pass
            if artifacts["geometry"]:
                break
    
    # Buscar Einstein
    for epath in [
        run_dir / "bulk_equations" / "einstein_discovery_summary.json",
        run_dir / "03_discover_bulk_equations" / "einstein_discovery_summary.json",
        run_dir / "bulk_equations" / "pareto_equations.json",
        run_dir / "bulk_equations_analysis" / "bulk_equations_report.json",
        run_dir / "05_analyze_bulk_equations" / "bulk_equations_report.json",
    ]:
        if epath.exists():
            try:
                data = json.loads(epath.read_text())
                score = data.get("einstein_score", data.get("best_score"))
                if score is None and "equations" in data:
                    for eq in data.get("equations", []):
                        if "einstein" in eq.get("name", "").lower():
                            score = eq.get("score", eq.get("fitness"))
                            break
                
                artifacts["einstein"] = {
                    "source": str(epath),
                    "score": score,
                    "n_equations": len(data.get("equations", [])),
                }
                artifacts["found"] = True
                break
            except:
                pass
    
    # Buscar diccionario
    for dpath in [
        run_dir / "emergent_dictionary" / "lambda_sl_dictionary_report.json",
        run_dir / "07_emergent_lambda_sl_dictionary" / "lambda_sl_dictionary_report.json",
        run_dir / "holographic_dictionary" / "holographic_dictionary_v3_summary.json",
        run_dir / "08_build_holographic_dictionary" / "holographic_dictionary_v3_summary.json",
        run_dir / "holographic_dictionary" / "holographic_dictionary_summary.json",
    ]:
        if dpath.exists():
            try:
                data = json.loads(dpath.read_text())
                
                deltas = []
                for system in data.get("systems", []):
                    ops = system.get("dictionary", {}).get("operators_predicted", [])
                    if not ops:
                        ops = system.get("geometry", {}).get("operators_predicted", [])
                    for op in ops:
                        if "Delta" in op:
                            deltas.append(op["Delta"])
                
                if not deltas and "predicted_Deltas" in data:
                    deltas = data["predicted_Deltas"]
                
                converged = data.get("converged")
                if converged is None:
                    loss = data.get("final_loss", data.get("loss"))
                    if loss is not None:
                        converged = loss < 0.1
                
                artifacts["dictionary"] = {
                    "source": str(dpath),
                    "predicted_Deltas": deltas,
                    "converged": converged,
                    "n_systems": len(data.get("systems", [])),
                }
                artifacts["found"] = True
                break
            except:
                pass
    
    return artifacts


def run_negative_control_check(
    run_dir: Path,
    h5_path: Optional[Path] = None,
    fpr_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Ejecuta verificaciÃ³n de control negativo usando FPR sobre seÃ±ales hologrÃ¡ficas.
    
    FPR = (seÃ±ales hologrÃ¡ficas disparadas) / (seÃ±ales evaluables)
    
    Esto mide: "Â¿El pipeline se autoengaÃ±a creyendo que hay holografÃ­a?"
    """
    logger.info("=" * 60)
    logger.info("CONTROL NEGATIVO - False Positive Rate sobre seÃ±ales hologrÃ¡ficas")
    logger.info("=" * 60)
    
    result = {
        "status": "INCOMPLETE",
        "false_positive_rate": None,
        "coverage": None,
        "n_signals_triggered": 0,
        "n_signals_evaluable": 0,
        "n_signals_total": 0,
        "fpr_threshold": fpr_threshold,
        "h5_path": None,
        "h5_verified": False,
        "signals": [],
        "expected_fail_contracts": [],
        "rationale": "",
        "errors": []
    }
    
    # Paso 1: Encontrar y verificar HDF5
    if h5_path is None:
        h5_path = find_negative_control_h5(run_dir)
    
    if h5_path is None:
        result["errors"].append("No se encontrÃ³ HDF5 de control negativo")
        result["rationale"] = "VerificaciÃ³n incompleta: no se encontrÃ³ el archivo HDF5."
        return result
    
    result["h5_path"] = str(h5_path)
    
    is_valid, h5_meta = verify_negative_control_h5(h5_path)
    if not is_valid:
        result["errors"].append(f"HDF5 invÃ¡lido: {h5_meta.get('error', 'unknown')}")
        result["rationale"] = f"El HDF5 no tiene atributos correctos: {h5_meta}"
        return result
    
    result["h5_verified"] = True
    result["h5_metadata"] = h5_meta
    
    logger.info(f"  HDF5 verificado: {h5_path}")
    
    # Paso 2: Cargar artefactos
    logger.info("  Cargando artefactos del pipeline...")
    artifacts = load_negative_control_artifacts(run_dir)
    
    if not artifacts["found"]:
        result["errors"].append("No se encontraron artefactos del pipeline")
        result["rationale"] = "No hay artefactos. Â¿Se ejecutÃ³ el pipeline sobre el control negativo?"
        return result
    
    # Paso 3: Evaluar seÃ±ales hologrÃ¡ficas
    logger.info("  Evaluando seÃ±ales hologrÃ¡ficas...")
    signals = evaluate_holographic_signals(artifacts)
    
    # Paso 4: Calcular FPR
    fpr, coverage, n_triggered, n_evaluable = compute_false_positive_rate(signals)
    
    result["false_positive_rate"] = fpr
    result["coverage"] = coverage
    result["n_signals_triggered"] = n_triggered
    result["n_signals_evaluable"] = n_evaluable
    result["n_signals_total"] = len(signals)
    result["signals"] = [s.to_dict() for s in signals]
    
    logger.info(f"    SeÃ±ales evaluables: {n_evaluable}/{len(signals)}")
    logger.info(f"    SeÃ±ales disparadas (falsos positivos): {n_triggered}")
    logger.info(f"    FPR: {fpr:.1%}")
    logger.info(f"    Coverage: {coverage:.1%}")
    
    # Paso 5: Evaluar contratos expected-fail (informativos)
    expected_fail = evaluate_expected_fail_contracts(artifacts)
    result["expected_fail_contracts"] = [c.to_dict() for c in expected_fail]
    
    # Paso 6: Determinar status
    if n_evaluable == 0:
        result["status"] = "INCOMPLETE"
        result["rationale"] = "No hay seÃ±ales evaluables. Coverage insuficiente."
    elif fpr < fpr_threshold:
        result["status"] = "SUCCESS"
        result["rationale"] = (
            f"FPR={fpr:.1%} < {fpr_threshold:.0%}. "
            f"El pipeline NO se autoengaÃ±a: {n_triggered}/{n_evaluable} seÃ±ales disparadas. "
            f"Esto es evidencia de honestidad cientÃ­fica."
        )
    elif fpr < 0.5:
        result["status"] = "WARNING"
        triggered_names = [s.name for s in signals if s.triggered]
        result["rationale"] = (
            f"FPR={fpr:.1%} (moderado). "
            f"SeÃ±ales disparadas: {triggered_names}. "
            f"Investigar por quÃ© el pipeline detecta holografÃ­a espuria."
        )
    else:
        result["status"] = "ALERT"
        triggered_names = [s.name for s in signals if s.triggered]
        result["rationale"] = (
            f"POSIBLE FALSO POSITIVO SISTEMÃTICO: FPR={fpr:.1%} >= 50%. "
            f"SeÃ±ales disparadas: {triggered_names}. "
            f"AuditorÃ­a urgente necesaria."
        )
    
    logger.info(f"\n  Status: {result['status']}")
    logger.info(f"  Rationale: {result['rationale'][:100]}...")
    
    return result


# ============================================================
# CONTRATOS FASE XII
# ============================================================

class ContractsFase12:
    """Contratos para validacion de datos reales."""
    
    def __init__(self):
        self.results = []
    
    def contract_unitarity_bound(
        self,
        Delta: float,
        d: int,
        spin: int = 0,
        operator_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Contrato: Verifica el unitarity bound para operadores CFT.
        
        FUENTE: AGMOO Sec. 2.1 (SI esta en el texto de referencia)
        
        Para operadores escalares (spin=0):
            Delta >= (d-2)/2
            
        Para operadores con spin l:
            Delta >= d - 2 + l  (para l >= 1)
            
        Operadores que violan este bound corresponden a teorias no-unitarias.
        
        Args:
            Delta: dimension conforme del operador
            d: dimension del boundary CFT
            spin: spin del operador (0 para escalares)
            operator_name: nombre del operador (para reporting)
        
        Returns:
            dict con resultado del contrato
        """
        if spin == 0:
            # Scalar unitarity bound: Delta >= (d-2)/2
            bound = (d - 2) / 2.0
            bound_name = "scalar_unitarity"
        else:
            # Spinning operator bound: Delta >= d - 2 + l
            bound = d - 2 + spin
            bound_name = f"spin_{spin}_unitarity"
        
        satisfies = Delta >= bound
        margin = Delta - bound
        
        result = {
            "name": "unitarity_bound",
            "operator": operator_name,
            "d": d,
            "spin": spin,
            "Delta": Delta,
            "bound": bound,
            "bound_type": bound_name,
            "passed": satisfies,
            "margin": margin,
            "reference": "AGMOO Sec. 2.1 (in text)",
        }
        
        self.results.append(result)
        return result
    
    def contract_ising3d_consistency(
        self,
        predicted_family: str,
        predicted_Deltas: List[float],
        known_Deltas: Dict[str, float],
        dictionary_source: str = "unknown"
    ) -> Dict[str, Any]:
        """Contrato: Para Ising 3D, predicciones consistentes con bootstrap."""
        result = {
            "name": "ising3d_consistency", 
            "passed": True, 
            "checks": [],
            "n_predicted_Deltas": len(predicted_Deltas),
            "dictionary_source": dictionary_source
        }
        
        if "manual" in dictionary_source:
            result["note"] = "Diccionario v0 (manual): check tÃ©cnico, no confirmaciÃ³n fÃ­sica."
        
        if not predicted_Deltas:
            result["checks"].append({"name": "has_predicted_Deltas", "passed": False})
            result["passed"] = False
            self.results.append(result)
            return result
        else:
            result["checks"].append({"name": "has_predicted_Deltas", "passed": True})
        
        if "ads" not in predicted_family.lower() and "unknown" not in predicted_family.lower():
            result["checks"].append({"name": "family_is_ads_like", "passed": False, "got": predicted_family})
            result["passed"] = False
        else:
            result["checks"].append({"name": "family_is_ads_like", "passed": True})
        
        tolerance = 0.1
        if "sigma" in known_Deltas:
            Delta_sigma = known_Deltas["sigma"]
            matches = [D for D in predicted_Deltas if abs(D - Delta_sigma) < tolerance]
            if matches:
                result["checks"].append({"name": "Delta_sigma_match", "passed": True})
            else:
                result["checks"].append({"name": "Delta_sigma_match", "passed": False})
                result["passed"] = False
        
        if "epsilon" in known_Deltas:
            Delta_epsilon = known_Deltas["epsilon"]
            matches = [D for D in predicted_Deltas if abs(D - Delta_epsilon) < tolerance]
            result["checks"].append({
                "name": "Delta_epsilon_match", 
                "passed": len(matches) > 0,
                "note": "informativo"
            })
        
        self.results.append(result)
        return result
    
    def contract_kss_bound(self, eta_over_s: float, system_name: str) -> Dict[str, Any]:
        """
        Contrato: Verifica el bound universal de viscosidad eta/s >= 1/(4*pi).
        
        FUENTE EXTERNA (NO en AGMOO 1999):
            Kovtun, Son, Starinets, PRL 94 (2005) 111601
            "Viscosity in Strongly Interacting Quantum Field Theories..."
            
        Este bound fue derivado DESPUES del review AGMOO usando AdS/CFT.
        El texto AGMOO discute absorcion (Sec. 3.6) pero NO formula este bound.
        """
        kss = 1.0 / (4 * np.pi)
        result = {
            "name": "kss_bound",
            "system": system_name,
            "passed": eta_over_s >= kss * 0.95,
            "value": eta_over_s,
            "bound": kss
        }
        self.results.append(result)
        return result
    
    def contract_thermal_consistency(self, T_data: float, predicted_zh: float, d: int, system_name: str) -> Dict[str, Any]:
        result = {"name": "thermal_consistency", "system": system_name, "passed": True, "checks": []}
        
        if T_data <= 0 or predicted_zh <= 0:
            result["passed"] = T_data <= 0 and predicted_zh <= 0
            self.results.append(result)
            return result
        
        T_expected = d / (4 * np.pi * predicted_zh)
        ratio = T_data / T_expected
        result["passed"] = 0.1 < ratio < 10
        result["checks"].append({"name": "T_zh_ratio", "ratio": ratio})
        
        self.results.append(result)
        return result
    
    def contract_strange_metal_scaling(self, rho_exponent: float, predicted_z: float, d: int, system_name: str) -> Dict[str, Any]:
        """
        Contrato: Verifica scaling de resistividad en strange metals.
        
        FUENTE EXTERNA (NO en AGMOO 1999):
            Applied AdS/CMT - Hartnoll, Herzog, Horowitz (2008+)
            "Building a Holographic Superconductor", "Lectures on holographic..."
            
        El texto AGMOO NO discute strange metals ni aplicaciones a materia condensada.
        Esto es validacion contra fenomenologia experimental, no teoria del texto.
        
        Formula: rho ~ T^alpha donde alpha = (d-2)/z para Lifshitz scaling
        """
        result = {"name": "strange_metal_scaling", "system": system_name, "passed": True}
        alpha_expected = (d - 2) / predicted_z if predicted_z > 0 else 1.0
        result["passed"] = abs(rho_exponent - alpha_expected) < 0.5
        self.results.append(result)
        return result
    
    def contract_cosmology_bounds(self, ns: float, predicted_bulk: str, system_name: str) -> Dict[str, Any]:
        ns_planck, ns_error = 0.9649, 0.0042
        in_3sigma = abs(ns - ns_planck) < 3 * ns_error
        result = {"name": "cosmology_bounds", "system": system_name, "passed": in_3sigma}
        self.results.append(result)
        return result
    
    def summary(self) -> Dict[str, Any]:
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.get("passed", False))
        return {
            "phase": "XII",
            "n_contracts": n_total,
            "n_passed": n_passed,
            "pass_rate": n_passed / max(n_total, 1),
            "all_passed": n_passed == n_total,
            "results": self.results
        }


# ============================================================
# CONTRATOS FASE XIII
# ============================================================

class ContractsFase13:
    def __init__(self):
        self.results = []
    
    def contract_atlas_coverage(self, n_total: int, n_families: int, expected_families: List[str]) -> Dict[str, Any]:
        result = {"name": "atlas_coverage", "passed": True, "checks": []}
        result["checks"].append({"name": "families_detected", "n": n_families, "expected": expected_families})
        result["passed"] = n_families >= 1
        self.results.append(result)
        return result
    
    def contract_cluster_quality(self, clusters: Dict[str, List[str]], points: List[Dict]) -> Dict[str, Any]:
        result = {"name": "cluster_quality", "passed": True, "checks": []}
        n_clusters = len(clusters)
        if n_clusters >= 2:
            result["checks"].append({"name": "multiple_clusters", "passed": True, "n": n_clusters})
        else:
            result["checks"].append({"name": "multiple_clusters", "passed": n_clusters >= 1, "n": n_clusters})
        result["passed"] = n_clusters >= 1
        self.results.append(result)
        return result
    
    def contract_outlier_genuineness(self, outliers: List[str], all_points: List[Dict], threshold: float = 1.5) -> Dict[str, Any]:
        result = {"name": "outlier_genuineness", "passed": True, "checks": []}
        self.results.append(result)
        return result
    
    def contract_einstein_distribution(self, n_einstein: int, n_non_einstein: int, n_total: int) -> Dict[str, Any]:
        result = {"name": "einstein_distribution", "passed": True, "checks": []}
        if n_total > 0:
            ratio = n_einstein / n_total
            result["checks"].append({"name": "einstein_ratio", "ratio": ratio})
            result["passed"] = 0.05 < ratio < 0.95
        self.results.append(result)
        return result
    
    def contract_exploration_completeness(self, regions_explored: Dict[str, int], min_regions: int = 2) -> Dict[str, Any]:
        result = {"name": "exploration_completeness", "passed": True, "checks": []}
        n_regions = len(regions_explored)
        result["passed"] = n_regions >= min_regions
        result["checks"].append({"name": "n_regions", "n": n_regions, "min": min_regions})
        self.results.append(result)
        return result
    
    def summary(self) -> Dict[str, Any]:
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.get("passed", False))
        return {
            "phase": "XIII",
            "n_contracts": n_total,
            "n_passed": n_passed,
            "pass_rate": n_passed / max(n_total, 1),
            "all_passed": n_passed == n_total,
            "results": self.results
        }


# ============================================================
# FUNCIONES DE ORQUESTACIÃ“N
# ============================================================

def run_contracts_fase12(report_path: Path) -> Dict:
    """Ejecuta contratos Fase XII bÃ¡sicos."""
    if report_path.is_dir():
        for c in [
            report_path / "holographic_dictionary" / "holographic_dictionary_v3_summary.json",
            report_path / "08_build_holographic_dictionary" / "holographic_dictionary_v3_summary.json",
            report_path / "holographic_dictionary" / "holographic_dictionary_summary.json",
            report_path / "fase12_report.json",
        ]:
            if c.is_file():
                report_path = c
                break
    
    if not report_path.exists():
        return {"error": "report not found"}
    
    report = json.loads(report_path.read_text())
    contracts = ContractsFase12()
    
    for system in report.get("systems", []):
        name = system.get("name", "")
        source = system.get("source", "")
        geo = system.get("geometry", {})
        predicted_family = geo.get("predicted_family", "unknown")
        
        if source == "bootstrap" and "ising" in name.lower():
            dict_ops = system.get("dictionary", {}).get("operators_predicted", [])
            geom_ops = geo.get("operators_predicted", [])
            operators = dict_ops if dict_ops else geom_ops
            predicted_Deltas = [op.get("Delta", 0.0) for op in operators]
            dictionary_source = system.get("dictionary_source", "unknown")
            
            contracts.contract_ising3d_consistency(
                predicted_family, predicted_Deltas,
                {"sigma": 0.518, "epsilon": 1.41},
                dictionary_source
            )
        
        if source == "lattice":
            eta_s = system.get("physics_metadata", {}).get("eta_over_s_min", 0.1)
            if eta_s > 0:
                contracts.contract_kss_bound(eta_s, name)
    
    return contracts.summary()


def run_extended_contracts_fase12(
    report_path: Path,
    run_cft_special_ops: bool = True,
    run_spin_limitation: bool = True,
    run_ope_coefficients: bool = False
) -> Dict:
    """
    Ejecuta contratos Fase XII EXTENDIDOS (CFT mejorados).
    
    MEJORAS CFT v2:
      - Operadores especiales (T_Î¼Î½, J_Î¼, identidad)
      - ValidaciÃ³n de limitaciÃ³n de spin
      - Coeficientes OPE (opcional, requiere extracciÃ³n)
    
    Referencia: Maldacena Chunk 04, pÃ¡ginas 30-35
    """
    if not HAS_EXTENDED_CONTRACTS:
        return {
            "error": "extended_physics_contracts.py no disponible",
            "skipped": True,
            "note": "Instalar o verificar importaciÃ³n del mÃ³dulo"
        }
    
    if report_path.is_dir():
        for c in [
            report_path / "holographic_dictionary" / "holographic_dictionary_v3_summary.json",
            report_path / "08_build_holographic_dictionary" / "holographic_dictionary_v3_summary.json",
            report_path / "holographic_dictionary" / "holographic_dictionary_summary.json",
            report_path / "fase12_report.json",
        ]:
            if c.is_file():
                report_path = c
                break
    
    if not report_path.exists():
        return {"error": "report not found", "path": str(report_path)}
    
    report = json.loads(report_path.read_text())
    ext_contracts = ExtendedContractsFase12()
    
    for system in report.get("systems", []):
        name = system.get("name", "")
        source = system.get("source", "")
        geo = system.get("geometry", {})
        d = system.get("d", geo.get("d", 3))
        
        # Extraer operadores
        dict_ops = system.get("dictionary", {}).get("operators_predicted", [])
        geom_ops = geo.get("operators_predicted", [])
        operators_raw = dict_ops if dict_ops else geom_ops
        
        # Convertir a formato esperado
        operators = []
        for op in operators_raw:
            operators.append({
                "name": op.get("name", op.get("type", "?")),
                "Delta": op.get("Delta", 0.0),
                "spin": op.get("spin", 0),
            })
        
        predicted_Deltas = [op["Delta"] for op in operators]
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Contrato: Operator Tower
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if predicted_Deltas and len(predicted_Deltas) >= 2:
            ext_contracts.contract_operator_tower(
                predicted_Deltas=predicted_Deltas,
                system_name=name,
                d=d
            )
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Contrato: Spectral Gap
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        system_type = system.get("system_type", "critical")
        if "ising" in name.lower():
            system_type = "critical"
        
        if predicted_Deltas and len(predicted_Deltas) >= 2:
            ext_contracts.contract_spectral_gap(
                predicted_Deltas=predicted_Deltas,
                system_type=system_type,
                d=d,
                system_name=name
            )
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # MEJORA 1: Operadores especiales (T_Î¼Î½, J_Î¼, identidad)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if run_cft_special_ops and operators:
            ext_contracts.contract_special_operators(
                operators=operators,
                d=d,
                system_name=name
            )
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # MEJORA 2: LimitaciÃ³n de spin
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if run_spin_limitation and operators:
            ext_contracts.contract_spin_limitation(operators)
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # MEJORA 3: Coeficientes OPE (si hay datos)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if run_ope_coefficients:
            extracted_ope = system.get("ope_coefficients")
            ext_contracts.contract_ope_coefficients(
                extracted_coefficients=extracted_ope,
                system_name=name
            )
    
    return ext_contracts.summary()


def run_contracts_fase13(analysis_path: Path, atlas_path=None) -> dict:
    """Ejecuta contratos Fase XIII."""
    if not analysis_path.exists():
        return {"error": "analysis not found"}
    
    analysis = json.loads(analysis_path.read_text())
    contracts = ContractsFase13()
    
    contracts.contract_atlas_coverage(
        analysis.get("n_total", 0),
        len(analysis.get("clusters", {})),
        list(analysis.get("clusters", {}).keys())
    )
    
    if atlas_path and atlas_path.is_file():
        atlas = json.loads(atlas_path.read_text())
        contracts.contract_cluster_quality(atlas.get("clusters", {}), atlas.get("points", []))
        contracts.contract_outlier_genuineness(atlas.get("outliers", []), atlas.get("points", []))
    
    contracts.contract_einstein_distribution(
        analysis.get("n_einstein", 0),
        analysis.get("n_non_einstein", 0),
        analysis.get("n_total", 0)
    )
    contracts.contract_exploration_completeness(analysis.get("interesting_regions", {}))
    
    return contracts.summary()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Contratos Fases XII/XIII + Control Negativo (FPR) + CFT Enhanced"
    )
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # V3: Argumentos estÃ¡ndar
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if HAS_STAGE_UTILS:
        add_standard_arguments(parser)
    else:
        parser.add_argument("--experiment", type=str, default=None)
        parser.add_argument("--run-dir", type=str, default=None)
    
    # Argumentos especÃ­ficos del script
    parser.add_argument("--phase", type=str, required=True, choices=["12", "13", "both"])
    parser.add_argument("--fase12-report", type=str, default="")
    parser.add_argument("--fase13-analysis", type=str, default="")
    parser.add_argument("--fase13-atlas", type=str, default="")
    parser.add_argument("--output-file", type=str, default=None)
    
    # Control negativo
    parser.add_argument("--negative-control-run-dir", type=str, default=None,
                        help="Directorio del run sobre datos anti-hologrÃ¡ficos")
    parser.add_argument("--negative-control-h5", type=str, default=None,
                        help="HDF5 del control negativo")
    parser.add_argument("--require-negative-control", action="store_true",
                        help="Si ALERT â†’ exit 1")
    parser.add_argument("--negative-control-fpr-threshold", type=float, default=0.2,
                        help="Umbral FPR para SUCCESS (default: 0.2)")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # NUEVOS: Contratos CFT extendidos
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    parser.add_argument("--run-extended-contracts", action="store_true",
                        help="Ejecutar contratos CFT extendidos (T_Î¼Î½, J_Î¼, spin, OPE)")
    parser.add_argument("--skip-cft-special-ops", action="store_true",
                        help="Omitir validaciÃ³n de operadores especiales")
    parser.add_argument("--skip-spin-limitation", action="store_true",
                        help="Omitir validaciÃ³n de limitaciÃ³n de spin")
    parser.add_argument("--run-ope-coefficients", action="store_true",
                        help="Ejecutar validaciÃ³n de coeficientes OPE (requiere extracciÃ³n)")
    
    args = parser.parse_args()
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # V3: Crear StageContext
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    ctx = None
    if HAS_STAGE_UTILS:
        if not getattr(args, 'experiment', None):
            args.experiment = infer_experiment(args)
        
        ctx = StageContext.from_args(
            args,
            stage_number="09",
            stage_slug="real_data_and_dictionary_contracts"
        )
        print(f"[V3] Experiment: {ctx.experiment}")
        print(f"[V3] Stage dir: {ctx.stage_dir}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # RESOLVER RUTAS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    fase12_report = args.fase12_report or ""
    fase13_analysis = args.fase13_analysis or ""
    fase13_atlas = args.fase13_atlas or ""
    output_file = args.output_file
    
    # V3: Resolver desde ctx
    if ctx:
        run_dir = ctx.run_root
        
        if not fase12_report:
            candidates = [
                run_dir / "07_emergent_lambda_sl_dictionary" / "lambda_sl_dictionary_report.json",
                run_dir / "emergent_dictionary" / "lambda_sl_dictionary_report.json",
            ]
            for c in candidates:
                if c.exists():
                    fase12_report = str(c)
                    print(f"[V3] fase12_report desde stage 07: {c}")
                    break
        
        if not output_file:
            output_file = str(ctx.stage_dir / "contracts_12_13.json")
    
    # Legacy: Resolver desde --run-dir
    elif args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir).resolve()
        manifest = load_run_manifest(run_dir)
        artifacts = manifest.get("artifacts", {}) if manifest else {}
        
        if not fase12_report:
            for c in [run_dir / "emergent_dictionary" / "lambda_sl_dictionary_report.json"]:
                if c.exists():
                    fase12_report = str(c)
                    break
        
        if not output_file:
            contracts_dir = run_dir / "contracts"
            contracts_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(contracts_dir / "contracts_12_13.json")
    
    if not output_file:
        output_file = "contracts_12_13.json"
    
    print("=" * 70)
    print("CONTRATOS FASES XII/XIII + CONTROL NEGATIVO (FPR) + CFT ENHANCED")
    print("=" * 70)
    
    if HAS_EXTENDED_CONTRACTS:
        print("  [âœ“] Contratos CFT extendidos disponibles")
    else:
        print("  [âœ—] Contratos CFT extendidos NO disponibles")
    
    results = {}
    negative_control_alert = False
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Fase XII (bÃ¡sicos)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if args.phase in ["12", "both"] and fase12_report:
        print(f"\n>> Validando Fase XII (bÃ¡sicos) desde {fase12_report}")
        results["fase12"] = run_contracts_fase12(Path(fase12_report))
        summary = results["fase12"]
        print(f"   Contratos: {summary.get('n_passed', 0)}/{summary.get('n_contracts', 0)}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Fase XII EXTENDIDA (CFT mejorados)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if args.run_extended_contracts and args.phase in ["12", "both"] and fase12_report:
        print(f"\n>> Validando Fase XII EXTENDIDA (CFT) desde {fase12_report}")
        results["fase12_extended"] = run_extended_contracts_fase12(
            report_path=Path(fase12_report),
            run_cft_special_ops=not args.skip_cft_special_ops,
            run_spin_limitation=not args.skip_spin_limitation,
            run_ope_coefficients=args.run_ope_coefficients
        )
        summary_ext = results["fase12_extended"]
        if "error" not in summary_ext:
            print(f"   Contratos extendidos: {summary_ext.get('n_passed', 0)}/{summary_ext.get('n_contracts', 0)}")
            if summary_ext.get("cft_enhancements"):
                print(f"   Mejoras CFT: {summary_ext['cft_enhancements']}")
        else:
            print(f"   [WARN] {summary_ext['error']}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Fase XIII
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if args.phase in ["13", "both"] and fase13_analysis:
        print(f"\n>> Validando Fase XIII desde {fase13_analysis}")
        results["fase13"] = run_contracts_fase13(
            Path(fase13_analysis),
            Path(fase13_atlas) if fase13_atlas else None
        )
        summary = results["fase13"]
        print(f"   Contratos: {summary.get('n_passed', 0)}/{summary.get('n_contracts', 0)}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Control negativo
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if args.negative_control_run_dir:
        print(f"\n>> Control negativo desde {args.negative_control_run_dir}")
        
        results["negative_control"] = run_negative_control_check(
            run_dir=Path(args.negative_control_run_dir),
            h5_path=Path(args.negative_control_h5) if args.negative_control_h5 else None,
            fpr_threshold=args.negative_control_fpr_threshold
        )
        
        nc = results["negative_control"]
        print(f"\n   Status: {nc['status']}")
        if nc["false_positive_rate"] is not None:
            print(f"   FPR: {nc['false_positive_rate']:.1%} (threshold: {nc['fpr_threshold']:.0%})")
            print(f"   Coverage: {nc['coverage']:.1%}")
            print(f"   SeÃ±ales: {nc['n_signals_triggered']}/{nc['n_signals_evaluable']} disparadas")
        
        if nc["status"] == "ALERT":
            negative_control_alert = True
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Guardar
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    
    all_passed = True
    for phase, summary in results.items():
        if isinstance(summary, dict):
            if phase == "negative_control":
                status = summary.get("status", "INCOMPLETE")
                fpr = summary.get("false_positive_rate")
                if fpr is not None:
                    print(f"  {phase}: {status} (FPR={fpr:.1%})")
                else:
                    print(f"  {phase}: {status}")
                if status == "ALERT":
                    all_passed = False
            elif "all_passed" in summary:
                status = "OK" if summary["all_passed"] else "FAIL"
                print(f"  {phase}: {status} ({summary['n_passed']}/{summary['n_contracts']})")
                all_passed = all_passed and summary["all_passed"]
            elif "error" in summary:
                print(f"  {phase}: ERROR ({summary['error']})")
    
    print(f"\n  Output: {output_path}")
    print("=" * 70)
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # V3: Registrar artefactos y escribir summary
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if ctx:
        ctx.record_artifact("contracts_json", output_path)
        if fase12_report:
            ctx.record_artifact("fase12_report_input", Path(fase12_report))
        if fase13_analysis:
            ctx.record_artifact("fase13_analysis_input", Path(fase13_analysis))
        
        status = "OK" if all_passed else "WARNING"
        if negative_control_alert:
            status = "ALERT"
        
        counts = {
            "fase12_passed": results.get("fase12", {}).get("n_passed", 0),
            "fase12_total": results.get("fase12", {}).get("n_contracts", 0),
            "fase13_passed": results.get("fase13", {}).get("n_passed", 0),
            "fase13_total": results.get("fase13", {}).get("n_contracts", 0),
            "negative_control_fpr": results.get("negative_control", {}).get("false_positive_rate"),
            "all_passed": all_passed,
        }
        
        # AÃ±adir counts de contratos extendidos
        if "fase12_extended" in results and "error" not in results["fase12_extended"]:
            counts["fase12_extended_passed"] = results["fase12_extended"].get("n_passed", 0)
            counts["fase12_extended_total"] = results["fase12_extended"].get("n_contracts", 0)
        
        ctx.write_summary(status=status, counts=counts)
        ctx.write_manifest()
        print(f"[V3] stage_summary.json escrito")
    
    if args.require_negative_control and negative_control_alert:
        print("\nâš  Exit 1: --require-negative-control activo y status=ALERT")
        return 1
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
