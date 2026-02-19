#!/usr/bin/env python3
"""
00b_physics_sanity_checks.py — Validación Física POST-HOC para CUERDAS-Maldacena

╔══════════════════════════════════════════════════════════════════════════════╗
║  ADVERTENCIA: ESTE SCRIPT ES POST-HOC                                        ║
║                                                                              ║
║  Este script NO es un filtro de datos. Genera FLAGS informativos para       ║
║  auditoría. Los datos que no cumplan relaciones teóricas NO son rechazados. ║
║                                                                              ║
║  La separación entre validación IO (00) y validación física (00b) es        ║
║  DELIBERADA: garantiza que el pipeline de descubrimiento NO asume teoría.   ║
╚══════════════════════════════════════════════════════════════════════════════╝

PROPÓSITO:
    Contrastar datos generados por el pipeline con relaciones teóricas CONOCIDAS.
    Esto permite detectar:
    1. Anomalías numéricas (bugs en el solver)
    2. Sistemas genuinamente no-holográficos (señal real)
    3. Regímenes donde la teoría estándar no aplica (física nueva)

RELACIONES VERIFICADAS (todas post-hoc, solo como contraste):

    1. Relación masa-dimensión conforme (AdS/CFT)
       Δ = d/2 + √(d²/4 + m²R²)
       Fuente: AGMOO Review, Sección 3.1.2, Ecuación (3.14)
       
    2. Cota de estabilidad Breitenlohner-Freedman
       m²R² ≥ -(d/2)²
       Fuente: AGMOO Review, Sección 2.2.2, Ecuación (2.42)
       
    3. Límites de unitariedad CFT
       Δ ≥ (d-2)/2 para escalares
       Fuente: AGMOO Review, Sección 3.1.3

USO:
    python 00b_physics_sanity_checks.py --input-csv runs/bulk_eigenmodes/bulk_modes_dataset.csv
    python 00b_physics_sanity_checks.py --experiment mi_exp --output report.json

SALIDA:
    JSON con flags por sistema/modo. NUNCA rechaza datos.

Autor: CUERDAS-Maldacena Team
Versión: 1.0 (2025-12)
Contrato: POST-HOC ONLY — NO FILTERING
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTES TEÓRICAS (con citas exactas)
# =============================================================================

THEORY_REFERENCES = {
    "mass_dimension_relation": {
        "equation": "Δ = d/2 + √(d²/4 + m²R²)",
        "source": "AGMOO Review",
        "section": "3.1.2",
        "equation_number": "(3.14)",
        "notes": [
            "Esta es la rama Δ+ (dimensión mayor). Δ- = d/2 - √(...) es la alternativa.",
            "m²R² corresponde a lambda_sl en nuestra nomenclatura.",
            "La relación asume AdS puro. Geometrías deformadas pueden desviarse.",
        ],
    },
    "bf_bound": {
        "equation": "m²R² ≥ -(d/2)²",
        "source": "AGMOO Review",
        "section": "2.2.2",
        "equation_number": "(2.42)",
        "notes": [
            "La cota BF garantiza estabilidad del vacío AdS.",
            "Violaciones indican vacío inestable o error numérico.",
            "En d=4 (CFT 3d): m²R² ≥ -4",
        ],
    },
    "unitarity_bound": {
        "equation": "Δ ≥ (d-2)/2",
        "source": "AGMOO Review",
        "section": "3.1.3",
        "notes": [
            "Cota de unitariedad para operadores escalares en CFT.",
            "En d=4 (CFT 3d): Δ ≥ 1",
            "Operadores con Δ < cota violan unitariedad.",
        ],
    },
}


# =============================================================================
# Data classes para el reporte
# =============================================================================

@dataclass
class PhysicsFlag:
    """Un flag de verificación física (informativo, NO rechaza datos)."""
    
    check_name: str       # Nombre del check (e.g., "bf_bound")
    status: str           # "OK", "FLAG", "ANOMALY"
    message: str          # Descripción legible
    theory_ref: str       # Referencia teórica (e.g., "AGMOO Eq. 2.42")
    values: Dict[str, Any] = field(default_factory=dict)  # Valores numéricos relevantes
    caveats: List[str] = field(default_factory=list)  # Notas sobre por qué el flag puede ser esperado
    
    def is_flagged(self) -> bool:
        return self.status in ("FLAG", "ANOMALY")
    
    def has_caveats(self) -> bool:
        return len(self.caveats) > 0


@dataclass
class ModeCheck:
    """Resultado de verificación para un modo específico."""
    
    system_name: str
    mode_id: int
    family: str
    d: int
    lambda_sl: float
    Delta_UV: Optional[float]
    flags: List[PhysicsFlag] = field(default_factory=list)
    
    @property
    def has_flags(self) -> bool:
        return any(f.is_flagged() for f in self.flags)
    
    @property
    def n_flags(self) -> int:
        return sum(1 for f in self.flags if f.is_flagged())


@dataclass
class PhysicsReport:
    """Reporte completo de verificación física POST-HOC."""
    
    timestamp: str
    input_file: str
    total_modes: int
    modes_with_flags: int
    modes_ok: int
    mode_checks: List[ModeCheck] = field(default_factory=list)
    summary_by_check: Dict[str, Dict[str, int]] = field(default_factory=dict)
    theory_references: Dict[str, Any] = field(default_factory=dict)
    
    # Metadatos de honestidad
    is_post_hoc: bool = True
    rejects_data: bool = False
    purpose: str = "Contraste con teoría conocida para auditoría"


# =============================================================================
# Detección de régimen (para contextualizar flags)
# =============================================================================

def detect_regime_caveats(system_name: str, mode_id: int, family: str) -> List[str]:
    """
    Detecta si el sistema está en un régimen donde la relación Δ-λ estándar
    puede no aplicar. Devuelve lista de caveats informativos.
    
    La relación Δ = d/2 + √(d²/4 + m²R²) asume:
    - AdS puro (sin horizonte)
    - Ground state (modo fundamental)
    - Sin deformaciones
    
    Desviaciones son ESPERADAS en otros regímenes.
    """
    caveats = []
    name_lower = system_name.lower()
    
    # Temperatura finita / horizonte
    finite_T_indicators = ["tfinite", "t_finite", "schwarzschild", "horizon", "blackhole", "bh_"]
    if any(ind in name_lower for ind in finite_T_indicators):
        caveats.append(
            "Sistema con temperatura finita/horizonte: la relación Δ-λ estándar "
            "asume AdS puro sin horizonte. Desviaciones son físicamente esperadas."
        )
    
    # Modos excitados
    if mode_id > 0:
        caveats.append(
            f"Modo excitado (n={mode_id}): la relación simple Δ-λ aplica principalmente "
            "al ground state. Modos excitados tienen estructura espectral más compleja."
        )
    
    # Geometrías deformadas
    deformed_indicators = ["deformed", "lifshitz", "hvlf", "hyperscaling", "anisotropic"]
    if any(ind in name_lower for ind in deformed_indicators) or family.lower() in ["lifshitz", "hvlf", "hyperscaling", "deformed"]:
        caveats.append(
            f"Geometría deformada (family={family}): la relación Δ-λ estándar es "
            "específica para AdS puro. Geometrías con exponentes anómalos tienen "
            "relaciones modificadas."
        )
    
    # Dp-branas no conformales (p ≠ 3)
    if family.lower() == "dpbrane" or "brane" in name_lower:
        # D3-branas (p=3) son conformales (AdS₅), pero otras no
        if "d3brane" not in name_lower:
            caveats.append(
                f"Dp-brana (family={family}): solo las D3-branas tienen dual conformal. "
                "Otras Dp-branas tienen teorías de gauge no-conformales (AGMOO Sec. 6.1.3). "
                "La relación Δ-λ estándar no aplica directamente."
            )
    
    # Datos reales / desconocidos
    if family.lower() in ["unknown", "real", "ising3d", "ising_3d"]:
        caveats.append(
            f"Sistema de tipo '{family}': no hay garantía de que sea holográfico. "
            "Desviaciones de la relación Δ-λ pueden indicar física no-AdS/CFT."
        )
    
    return caveats


# =============================================================================
# Funciones de verificación física
# =============================================================================

def compute_delta_from_lambda(lambda_sl: float, d: int) -> Tuple[float, float]:
    """
    Calcula Δ+ y Δ- desde λ_sl usando la relación masa-dimensión.
    
    Δ± = d/2 ± √(d²/4 + λ_sl)
    
    Donde λ_sl = m²R² (autovalor Sturm-Liouville).
    
    Fuente: AGMOO Sección 3.1.2, Ecuación (3.14)
    
    Returns:
        (Delta_plus, Delta_minus)
    """
    discriminant = (d / 2.0) ** 2 + lambda_sl
    
    if discriminant < 0:
        # Discriminante negativo: Δ complejo (inestabilidad)
        return (np.nan, np.nan)
    
    sqrt_disc = np.sqrt(discriminant)
    delta_plus = d / 2.0 + sqrt_disc
    delta_minus = d / 2.0 - sqrt_disc
    
    return (delta_plus, delta_minus)


def check_bf_bound(lambda_sl: float, d: int) -> PhysicsFlag:
    """
    Verifica la cota Breitenlohner-Freedman: λ_sl ≥ -(d/2)²
    
    Fuente: AGMOO Sección 2.2.2, Ecuación (2.42)
    """
    bf_bound = -(d / 2.0) ** 2
    
    if lambda_sl >= bf_bound:
        return PhysicsFlag(
            check_name="bf_bound",
            status="OK",
            message=f"λ_sl = {lambda_sl:.4f} ≥ {bf_bound:.4f} (cota BF satisfecha)",
            theory_ref="AGMOO Sec. 2.2.2, Eq. (2.42)",
            values={"lambda_sl": lambda_sl, "bf_bound": bf_bound, "d": d},
        )
    else:
        margin = bf_bound - lambda_sl
        return PhysicsFlag(
            check_name="bf_bound",
            status="FLAG",
            message=f"λ_sl = {lambda_sl:.4f} < {bf_bound:.4f} (viola cota BF por {margin:.4f})",
            theory_ref="AGMOO Sec. 2.2.2, Eq. (2.42)",
            values={"lambda_sl": lambda_sl, "bf_bound": bf_bound, "violation": margin, "d": d},
        )


def check_mass_dimension_relation(
    lambda_sl: float, 
    Delta_UV: Optional[float], 
    d: int,
    system_name: str = "",
    mode_id: int = 0,
    family: str = "",
    tolerance: float = 0.1
) -> PhysicsFlag:
    """
    Verifica consistencia con la relación Δ = d/2 + √(d²/4 + m²R²).
    
    Fuente: AGMOO Sección 3.1.2, Ecuación (3.14)
    
    Args:
        lambda_sl: Autovalor Sturm-Liouville (= m²R²)
        Delta_UV: Dimensión conforme medida desde correladores
        d: Dimensión del bulk
        system_name: Nombre del sistema (para detectar régimen)
        mode_id: ID del modo (0=ground state)
        family: Familia del sistema
        tolerance: Tolerancia relativa para considerar "consistente"
    """
    if Delta_UV is None or np.isnan(Delta_UV):
        return PhysicsFlag(
            check_name="mass_dimension_relation",
            status="OK",  # No podemos verificar sin Δ
            message="Delta_UV no disponible, check no aplicable",
            theory_ref="AGMOO Sec. 3.1.2, Eq. (3.14)",
            values={"lambda_sl": lambda_sl, "Delta_UV": None, "d": d},
        )
    
    delta_plus, delta_minus = compute_delta_from_lambda(lambda_sl, d)
    
    if np.isnan(delta_plus):
        return PhysicsFlag(
            check_name="mass_dimension_relation",
            status="ANOMALY",
            message=f"Discriminante negativo: d²/4 + λ_sl = {(d/2)**2 + lambda_sl:.4f} < 0",
            theory_ref="AGMOO Sec. 3.1.2, Eq. (3.14)",
            values={"lambda_sl": lambda_sl, "Delta_UV": Delta_UV, "d": d, "discriminant": (d/2)**2 + lambda_sl},
        )
    
    # Verificar si Delta_UV coincide con Δ+ o Δ-
    error_plus = abs(Delta_UV - delta_plus) / max(abs(delta_plus), 1e-10)
    error_minus = abs(Delta_UV - delta_minus) / max(abs(delta_minus), 1e-10) if delta_minus > 0 else float('inf')
    
    min_error = min(error_plus, error_minus)
    matched_branch = "Δ+" if error_plus <= error_minus else "Δ-"
    expected = delta_plus if matched_branch == "Δ+" else delta_minus
    
    if min_error <= tolerance:
        return PhysicsFlag(
            check_name="mass_dimension_relation",
            status="OK",
            message=f"Δ_UV = {Delta_UV:.4f} ≈ {matched_branch} = {expected:.4f} (error {min_error*100:.1f}%)",
            theory_ref="AGMOO Sec. 3.1.2, Eq. (3.14)",
            values={
                "lambda_sl": lambda_sl, 
                "Delta_UV": Delta_UV, 
                "delta_plus": delta_plus,
                "delta_minus": delta_minus,
                "matched_branch": matched_branch,
                "relative_error": min_error,
                "d": d,
            },
        )
    else:
        # Detectar caveats para explicar por qué el flag puede ser esperado
        caveats = detect_regime_caveats(system_name, mode_id, family)
        
        return PhysicsFlag(
            check_name="mass_dimension_relation",
            status="FLAG",
            message=f"Δ_UV = {Delta_UV:.4f} difiere de Δ+ = {delta_plus:.4f} y Δ- = {delta_minus:.4f} (error mín {min_error*100:.1f}%)",
            theory_ref="AGMOO Sec. 3.1.2, Eq. (3.14)",
            values={
                "lambda_sl": lambda_sl, 
                "Delta_UV": Delta_UV, 
                "delta_plus": delta_plus,
                "delta_minus": delta_minus,
                "relative_error": min_error,
                "d": d,
            },
            caveats=caveats,
        )


def check_unitarity_bound(Delta_UV: Optional[float], d: int) -> PhysicsFlag:
    """
    Verifica la cota de unitariedad CFT: Δ ≥ (d-2)/2
    
    Fuente: AGMOO Sección 3.1.3
    """
    if Delta_UV is None or np.isnan(Delta_UV):
        return PhysicsFlag(
            check_name="unitarity_bound",
            status="OK",
            message="Delta_UV no disponible, check no aplicable",
            theory_ref="AGMOO Sec. 3.1.3",
            values={"Delta_UV": None, "d": d},
        )
    
    # d en el bulk = d_CFT + 1, así que d_CFT = d - 1
    # Cota de unitariedad para escalares: Δ ≥ (d_CFT - 2)/2 = (d - 3)/2
    # Pero usualmente se expresa como Δ ≥ (d-2)/2 donde d es la dim de la CFT
    # En nuestra convención, d es dim del bulk, así que la CFT tiene dim d-1
    d_cft = d - 1
    unitarity_bound = (d_cft - 2) / 2.0
    
    if Delta_UV >= unitarity_bound:
        return PhysicsFlag(
            check_name="unitarity_bound",
            status="OK",
            message=f"Δ = {Delta_UV:.4f} ≥ {unitarity_bound:.4f} (unitariedad OK)",
            theory_ref="AGMOO Sec. 3.1.3",
            values={"Delta_UV": Delta_UV, "unitarity_bound": unitarity_bound, "d": d, "d_cft": d_cft},
        )
    else:
        violation = unitarity_bound - Delta_UV
        return PhysicsFlag(
            check_name="unitarity_bound",
            status="FLAG",
            message=f"Δ = {Delta_UV:.4f} < {unitarity_bound:.4f} (viola unitariedad por {violation:.4f})",
            theory_ref="AGMOO Sec. 3.1.3",
            values={"Delta_UV": Delta_UV, "unitarity_bound": unitarity_bound, "violation": violation, "d": d, "d_cft": d_cft},
        )


def check_single_mode(row: pd.Series, tolerance: float = 0.1) -> ModeCheck:
    """Ejecuta todos los checks para un modo."""
    
    system_name = str(row.get("system_name", "unknown"))
    mode_id = int(row.get("mode_id", 0))
    family = str(row.get("family", "unknown"))
    d = int(row.get("d", 4))
    lambda_sl = float(row.get("lambda_sl", np.nan))
    Delta_UV = row.get("Delta_UV")
    if Delta_UV is not None and not np.isnan(Delta_UV):
        Delta_UV = float(Delta_UV)
    else:
        Delta_UV = None
    
    mode_check = ModeCheck(
        system_name=system_name,
        mode_id=mode_id,
        family=family,
        d=d,
        lambda_sl=lambda_sl,
        Delta_UV=Delta_UV,
    )
    
    # Check 1: Cota BF
    if not np.isnan(lambda_sl):
        mode_check.flags.append(check_bf_bound(lambda_sl, d))
    
    # Check 2: Relación masa-dimensión
    if not np.isnan(lambda_sl):
        mode_check.flags.append(check_mass_dimension_relation(
            lambda_sl, Delta_UV, d, 
            system_name=system_name, 
            mode_id=mode_id, 
            family=family,
            tolerance=tolerance
        ))
    
    # Check 3: Unitariedad
    mode_check.flags.append(check_unitarity_bound(Delta_UV, d))
    
    return mode_check


# =============================================================================
# Pipeline principal
# =============================================================================

def run_physics_checks(
    input_csv: Path,
    tolerance: float = 0.1,
    verbose: bool = True,
) -> PhysicsReport:
    """
    Ejecuta verificaciones físicas POST-HOC sobre un dataset de modos.
    
    IMPORTANTE: Este análisis es POST-HOC. No filtra ni rechaza datos.
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("CUERDAS-Maldacena — Physics Sanity Checks (POST-HOC)")
        print("=" * 70)
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║  NOTA: Este análisis es POST-HOC. NO rechaza datos.             ║")
        print("║  Los flags son informativos para auditoría.                      ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()
    
    # Leer CSV
    df = pd.read_csv(input_csv)
    
    if verbose:
        print(f"Input: {input_csv}")
        print(f"Total modos: {len(df)}")
        print()
    
    # Ejecutar checks
    mode_checks = []
    for _, row in df.iterrows():
        mode_check = check_single_mode(row, tolerance)
        mode_checks.append(mode_check)
    
    # Contar resultados
    modes_with_flags = sum(1 for mc in mode_checks if mc.has_flags)
    modes_ok = len(mode_checks) - modes_with_flags
    
    # Resumen por check
    summary_by_check = {}
    check_names = ["bf_bound", "mass_dimension_relation", "unitarity_bound"]
    for check_name in check_names:
        counts = {"OK": 0, "FLAG": 0, "ANOMALY": 0}
        for mc in mode_checks:
            for flag in mc.flags:
                if flag.check_name == check_name:
                    counts[flag.status] = counts.get(flag.status, 0) + 1
        summary_by_check[check_name] = counts
    
    # Crear reporte
    report = PhysicsReport(
        timestamp=datetime.now().isoformat(),
        input_file=str(input_csv),
        total_modes=len(mode_checks),
        modes_with_flags=modes_with_flags,
        modes_ok=modes_ok,
        mode_checks=mode_checks,
        summary_by_check=summary_by_check,
        theory_references=THEORY_REFERENCES,
    )
    
    if verbose:
        print("─" * 70)
        print("RESUMEN")
        print("─" * 70)
        print(f"  Total modos:       {report.total_modes}")
        print(f"  Modos OK:          {report.modes_ok}")
        print(f"  Modos con flags:   {report.modes_with_flags}")
        print()
        print("  Por verificación:")
        for check_name, counts in summary_by_check.items():
            print(f"    {check_name}:")
            print(f"      OK: {counts.get('OK', 0)}, FLAG: {counts.get('FLAG', 0)}, ANOMALY: {counts.get('ANOMALY', 0)}")
        
        # Estadísticas de caveats (flags esperados vs inesperados)
        flags_with_caveats = 0
        flags_without_caveats = 0
        for mc in mode_checks:
            for flag in mc.flags:
                if flag.is_flagged():
                    if flag.has_caveats():
                        flags_with_caveats += 1
                    else:
                        flags_without_caveats += 1
        
        if flags_with_caveats > 0 or flags_without_caveats > 0:
            print()
            print("  Análisis de flags:")
            print(f"    Con contexto (desviación esperada):    {flags_with_caveats}")
            print(f"    Sin contexto (investigar):             {flags_without_caveats}")
        print()
        
        # Mostrar algunos ejemplos de flags
        flagged_modes = [mc for mc in mode_checks if mc.has_flags]
        if flagged_modes:
            print("─" * 70)
            print("EJEMPLOS DE FLAGS (primeros 5)")
            print("─" * 70)
            for mc in flagged_modes[:5]:
                print(f"\n  {mc.system_name} / mode {mc.mode_id} (family={mc.family}, d={mc.d}):")
                print(f"    λ_sl = {mc.lambda_sl:.6f}, Δ_UV = {mc.Delta_UV}")
                for flag in mc.flags:
                    if flag.is_flagged():
                        print(f"    [{flag.status}] {flag.check_name}: {flag.message}")
                        print(f"           Ref: {flag.theory_ref}")
                        # Mostrar caveats si existen
                        if flag.has_caveats():
                            print(f"           ⚠ CONTEXTO (desviación puede ser esperada):")
                            for caveat in flag.caveats:
                                # Wrap long caveats
                                wrapped = caveat[:80] + "..." if len(caveat) > 80 else caveat
                                print(f"             • {wrapped}")
        
        print()
        print("=" * 70)
        print("[POST-HOC] Verificación completada. NINGÚN dato fue rechazado.")
        print("=" * 70)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verificación Física POST-HOC para CUERDAS-Maldacena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANTE:
  Este script es POST-HOC. No rechaza datos.
  Los flags son informativos para auditoría.
  
  La separación entre validación IO (00) y validación física (00b) es
  DELIBERADA: garantiza que el pipeline de descubrimiento NO asume teoría.

Ejemplos:
  python 00b_physics_sanity_checks.py --input-csv runs/bulk_eigenmodes/bulk_modes_dataset.csv
  python 00b_physics_sanity_checks.py --input-csv data.csv --output report.json --tolerance 0.05
        """
    )
    
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="CSV con columnas: system_name, family, d, mode_id, lambda_sl, Delta_UV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Archivo de salida para el reporte JSON (opcional)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Tolerancia relativa para relación masa-dimensión (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprimir output a consola",
    )
    
    args = parser.parse_args()
    
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        print(f"ERROR: No existe el archivo {input_csv}")
        sys.exit(1)
    
    report = run_physics_checks(
        input_csv=input_csv,
        tolerance=args.tolerance,
        verbose=not args.quiet,
    )
    
    # Guardar reporte si se especifica
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir a dict serializable
        report_dict = {
            "timestamp": report.timestamp,
            "input_file": report.input_file,
            "total_modes": report.total_modes,
            "modes_with_flags": report.modes_with_flags,
            "modes_ok": report.modes_ok,
            "is_post_hoc": report.is_post_hoc,
            "rejects_data": report.rejects_data,
            "purpose": report.purpose,
            "summary_by_check": report.summary_by_check,
            "theory_references": report.theory_references,
            "mode_checks": [
                {
                    "system_name": mc.system_name,
                    "mode_id": mc.mode_id,
                    "family": mc.family,
                    "d": mc.d,
                    "lambda_sl": mc.lambda_sl,
                    "Delta_UV": mc.Delta_UV,
                    "has_flags": mc.has_flags,
                    "n_flags": mc.n_flags,
                    "flags": [asdict(f) for f in mc.flags],
                }
                for mc in report.mode_checks
            ],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        if not args.quiet:
            print(f"\nReporte guardado en: {output_path}")
    
    # Siempre exit 0 — NO rechazamos datos
    sys.exit(0)


if __name__ == "__main__":
    main()
