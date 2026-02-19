#!/usr/bin/env python3
# 05_analyze_bulk_equations.py
# CUERDAS - Bloque A: Geometria emergente (analisis de ecuaciones)
#
# OBJETIVO
#   Analizar la familia de ecuaciones de bulk descubiertas:
#     - Patrones universales vs especificos de familia.
#     - Dependencia en dimension d, z_dyn, theta, etc.
#     - Relacion entre complejidad, error y clasificacion fisica.
#
# ENTRADAS
#   - runs/<experiment>/03_discover_bulk_equations/einstein_discovery_summary.json
#
# SALIDAS
#   runs/<experiment>/05_analyze_bulk_equations/
#     bulk_equations_report.txt
#     bulk_equations_report.json
#     stage_summary.json
#
# RELACION CON OTROS SCRIPTS
#   - Depende de: 03_discover_bulk_equations.py
#   - No afecta a la tuberia de entrenamiento; es puramente analisis/diagnostico.
#
# ============================================================================
# NOTA SOBRE NOMENCLATURA Y TEORIA (actualizado 2024-12-29)
# ============================================================================
#
# Este script usa terminologia moderna de AdS/CMT (Applied AdS/CFT):
#   - "Lifshitz" geometries con exponente dinamico z
#   - "Hyperscaling violation" con parametro theta
#
# Esta nomenclatura (z, theta) es una PARAMETRIZACION FENOMENOLOGICA moderna
# que NO aparece explicitamente en el texto de referencia (AGMOO 1999).
#
# RELACION CON EL TEXTO DE REFERENCIA:
#   - El texto (Sec. 6.1.3) describe Dp-branas via funciones armonicas H(r)
#   - Las metricas Dp-brana tienen exponentes efectivos que PUEDEN mapearse
#     a (z, theta) pero el texto no usa esta parametrizacion
#   - AdS puro corresponde a z=1, theta=0 (Sec. 2.2, Eq. 2.23, 2.27)
#
# HONESTIDAD EPISTEMOLOGICA:
#   - Esta clasificacion es ANALISIS POST-HOC, no inyeccion de teoria
#   - Los exponentes z, theta se EXTRAEN de los nombres de archivo
#     (generados en 01_generate_sandbox_geometries.py)
#   - NO se usan para entrenar ni como loss functions
#   - Solo sirven para organizar y comparar resultados
#
# REFERENCIAS MODERNAS (no en AGMOO 1999):
#   - Lifshitz holography: Kachru, Liu, Mulligan (2008)
#   - Hyperscaling violation: Gouteraux, Kiritsis (2011)
#   - Estas son generalizaciones de los backgrounds Dp-brana del Cap. 6
#
# ============================================================================
#
# MIGRADO A V3: 2024-12-23
# DOCUMENTACION ACTUALIZADA: 2024-12-29

import json
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# V3 INFRASTRUCTURE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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

if not HAS_STAGE_UTILS:
    print("[WARN] stage_utils not available, running in legacy mode")

# Legacy imports (fallback)
try:
    from cuerdas_io import resolve_bulk_equations_dir, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


def parse_equation_coefficients(eq_str: str) -> Dict[str, float]:
    """
    Extrae coeficientes numericos de una ecuacion de PySR.
    
    Ejemplo: "((-20.000029 * square(x2)) - (9.999996 * x3))"
    â†’ {"square_x2": -20.0, "x3": -10.0}
    """
    coefficients = {}
    
    # Buscar patrones como "numero * termino"
    patterns = [
        (r'([-]?\d+\.?\d*)\s*\*\s*square\(x(\d)\)', 'square_x'),
        (r'([-]?\d+\.?\d*)\s*\*\s*x(\d)', 'x'),
        (r'([-]?\d+\.?\d*)\s*\*\s*cube\(x(\d)\)', 'cube_x'),
        (r'x(\d)\s*\*\s*([-]?\d+\.?\d*)', 'x'),  # Orden invertido
    ]
    
    for pattern, prefix in patterns:
        matches = re.findall(pattern, eq_str)
        for match in matches:
            if isinstance(match, tuple):
                coef, idx = match[0], match[1]
                try:
                    coefficients[f"{prefix}{idx}"] = float(coef)
                except:
                    pass
    
    # Buscar divisiones
    div_pattern = r'/\s*([-]?\d+\.?\d*)'
    div_matches = re.findall(div_pattern, eq_str)
    for i, coef in enumerate(div_matches):
        try:
            coefficients[f"div_{i}"] = float(coef)
        except:
            pass
    
    return coefficients


def analyze_equation_structure(eq_str: str) -> Dict[str, bool]:
    """Analiza que terminos aparecen en la ecuacion."""
    structure = {
        "has_dA_squared": "square(x2)" in eq_str or "x2 * x2" in eq_str,
        "has_d2A": "x3" in eq_str,
        "has_df": "x4" in eq_str,
        "has_d2f": "x5" in eq_str,
        "has_A": "x0" in eq_str and "x0)" not in eq_str.replace("x0 ", ""),
        "has_f": "x1" in eq_str,
        "has_cross_terms": ("x2" in eq_str and "x4" in eq_str) or ("x1" in eq_str and "x4" in eq_str),
        "complexity": eq_str.count("(") + eq_str.count("*") + eq_str.count("/")
    }
    return structure


def load_and_analyze(json_path: Path) -> Dict:
    """Carga resultados y analiza patrones."""
    
    with open(json_path) as f:
        data = json.load(f)
    
    results = {
        "by_family": defaultdict(list),
        "by_geometry": {},
        "patterns": {},
        "coefficients_vs_params": []
    }
    
    for geo in data["geometries"]:
        name = geo["name"]
        
        # ====================================================================
        # EXTRACCION POST-HOC de exponentes z y theta
        # ====================================================================
        # NOTA: z_dyn y theta se extraen de los NOMBRES de archivo, que fueron
        # generados en 01_generate_sandbox_geometries.py. Estos parametros:
        #   - NO se usaron en entrenamiento
        #   - NO afectan la regresion simbolica
        #   - Solo sirven para ORGANIZAR resultados post-hoc
        #
        # Nomenclatura moderna (post-1999):
        #   - z_dyn: exponente dinamico Lifshitz (z=1 para AdS, z>1 anisotropico)
        #   - theta: violacion de hyperscaling (theta=0 para AdS)
        #
        # Relacion con AGMOO: estas son generalizaciones fenomenologicas de
        # los backgrounds Dp-brana descritos en Sec. 6.1.3
        # ====================================================================
        z_dyn = 1.0
        theta = 0.0
        
        if "lifshitz_z" in name:
            match = re.search(r'z(\d+)p(\d+)', name)
            if match:
                z_dyn = float(f"{match.group(1)}.{match.group(2)}")
        
        if "hvlf" in name:
            z_match = re.search(r'z(\d+)p(\d+)', name)
            theta_match = re.search(r'theta(\d+)p(\d+)', name)
            if z_match:
                z_dyn = float(f"{z_match.group(1)}.{z_match.group(2)}")
            if theta_match:
                theta = float(f"{theta_match.group(1)}.{theta_match.group(2)}")
        
        if name == "ads5":
            z_dyn = 1.0
            theta = 0.0
        
        # Analizar ecuacion de R
        if "R_equation" in geo["results"]:
            R_eq = geo["results"]["R_equation"]
            eq_str = R_eq["equation"]
            r2 = R_eq["r2"]
            
            coeffs = parse_equation_coefficients(eq_str)
            structure = analyze_equation_structure(eq_str)
            
            # Determinar familia (clasificacion post-hoc basada en nombre)
            if "ads" in name and "hvlf" not in name and "lifshitz" not in name:
                family = "ads"
            elif "lifshitz" in name and "hvlf" not in name:
                family = "lifshitz"
            else:
                family = "hyperscaling"
            
            geo_result = {
                "name": name,
                "z_dyn": z_dyn,
                "theta": theta,
                "family": family,
                "R_equation": eq_str,
                "R_r2": r2,
                "coefficients": coeffs,
                "structure": structure,
                "validation": geo.get("validation", {})
            }
            
            results["by_geometry"][name] = geo_result
            results["by_family"][family].append(geo_result)
            results["coefficients_vs_params"].append({
                "z": z_dyn,
                "theta": theta,
                "family": family,
                "coeffs": coeffs,
                "r2": r2
            })
    
    return results


def find_universal_structure(results: Dict) -> Dict:
    """Busca estructura universal en las ecuaciones."""
    
    patterns = {
        "universal_terms": set(),
        "family_specific_terms": defaultdict(set),
        "coefficient_trends": {}
    }
    
    # Analizar que terminos aparecen en todas las geometrias
    all_structures = [g["structure"] for g in results["by_geometry"].values()]
    
    if all_structures:
        first = all_structures[0]
        for key in first:
            if key != "complexity":
                values = [s[key] for s in all_structures]
                if all(values):
                    patterns["universal_terms"].add(key)
    
    # Analizar por familia
    for family, geos in results["by_family"].items():
        if geos:
            structures = [g["structure"] for g in geos]
            for key in structures[0]:
                if key != "complexity":
                    values = [s[key] for s in structures]
                    if all(values):
                        patterns["family_specific_terms"][family].add(key)
    
    # Buscar tendencias en coeficientes
    data = results["coefficients_vs_params"]
    
    # Agrupar por familia y buscar dependencia en z
    for family in ["ads", "lifshitz", "hyperscaling"]:
        family_data = [d for d in data if d["family"] == family]
        if len(family_data) >= 2:
            z_values = [d["z"] for d in family_data]
            
            # Buscar coeficientes comunes
            common_coeffs = set(family_data[0]["coeffs"].keys())
            for d in family_data[1:]:
                common_coeffs &= set(d["coeffs"].keys())
            
            for coeff_name in common_coeffs:
                coeff_values = [d["coeffs"][coeff_name] for d in family_data]
                
                # Calcular correlacion con z
                if len(set(z_values)) > 1:
                    corr = np.corrcoef(z_values, coeff_values)[0, 1]
                    if not np.isnan(corr):
                        patterns["coefficient_trends"][f"{family}_{coeff_name}"] = {
                            "correlation_with_z": float(corr),
                            "values": list(zip(z_values, coeff_values))
                        }
    
    return patterns


def generate_report(results: Dict, patterns: Dict) -> str:
    """Genera reporte de analisis."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("ANÃLISIS DE ECUACIONES DESCUBIERTAS POR PySR")
    lines.append("=" * 70)
    
    # Resumen por familia
    lines.append("\n## ECUACIONES POR FAMILIA\n")
    
    for family in ["ads", "lifshitz", "hyperscaling"]:
        geos = results["by_family"].get(family, [])
        if geos:
            lines.append(f"\n### {family.upper()} ({len(geos)} geometrias)")
            lines.append("-" * 50)
            
            for g in geos[:3]:  # Mostrar maximo 3
                lines.append(f"\n  {g['name']} (z={g['z_dyn']}, Î¸={g['theta']})")
                lines.append(f"  R = {g['R_equation'][:60]}...")
                lines.append(f"  R^2 = {g['R_r2']:.6f}")
    
    # Terminos universales
    lines.append("\n\n## ESTRUCTURA UNIVERSAL")
    lines.append("-" * 50)
    
    if patterns["universal_terms"]:
        lines.append("\nTerminos presentes en TODAS las geometrias:")
        for term in patterns["universal_terms"]:
            lines.append(f"  âœ“ {term}")
    
    # Terminos especificos por familia
    lines.append("\n\n## TÃ‰RMINOS ESPECÃFICOS POR FAMILIA")
    lines.append("-" * 50)
    
    for family, terms in patterns["family_specific_terms"].items():
        extra_terms = terms - patterns["universal_terms"]
        if extra_terms:
            lines.append(f"\n{family}:")
            for term in extra_terms:
                lines.append(f"  + {term}")
    
    # Tendencias en coeficientes
    if patterns["coefficient_trends"]:
        lines.append("\n\n## TENDENCIAS EN COEFICIENTES")
        lines.append("-" * 50)
        
        for name, trend in patterns["coefficient_trends"].items():
            corr = trend["correlation_with_z"]
            if abs(corr) > 0.5:
                direction = "aumenta" if corr > 0 else "disminuye"
                lines.append(f"\n  {name}: {direction} con z (r={corr:.2f})")
    
    # Conclusiones
    lines.append("\n\n## CONCLUSIONES")
    lines.append("=" * 70)
    
    # Verificar si hay estructura diferente
    ads_geos = results["by_family"].get("ads", [])
    lifshitz_geos = results["by_family"].get("lifshitz", [])
    hvlf_geos = results["by_family"].get("hyperscaling", [])
    
    if ads_geos and lifshitz_geos:
        ads_struct = ads_geos[0]["structure"] if ads_geos else {}
        lif_struct = lifshitz_geos[0]["structure"] if lifshitz_geos else {}
        
        if ads_struct != lif_struct:
            lines.append("\nâœ“ Las ecuaciones para Lifshitz son DIFERENTES a AdS")
            lines.append("  Esto indica fisica genuinamente distinta")
    
    if hvlf_geos:
        hvlf_with_cross = [g for g in hvlf_geos if g["structure"].get("has_cross_terms")]
        if hvlf_with_cross:
            lines.append(f"\nâ€¢ {len(hvlf_with_cross)} geometrias hyperscaling tienen terminos cruzados")
            lines.append("  Esto indica acoplamiento materia-geometria no trivial")
    
    # R^2 promedio
    all_r2 = [g["R_r2"] for g in results["by_geometry"].values()]
    if all_r2:
        avg_r2 = np.mean(all_r2)
        lines.append(f"\nâ€¢ R^2 promedio: {avg_r2:.6f}")
        if avg_r2 > 0.999:
            lines.append("  Las ecuaciones descubiertas son muy precisas")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analiza ecuaciones descubiertas")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # V3: Argumentos estandar
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if HAS_STAGE_UTILS:
        add_standard_arguments(parser)
    else:
        parser.add_argument("--experiment", type=str, default=None)
        parser.add_argument("--run-dir", type=str, default=None)
    
    # Argumentos especificos de este script (legacy compatibility)
    parser.add_argument("--input", type=str, default=None,
                        help="Archivo einstein_discovery_summary.json (legacy)")
    parser.add_argument("--output", type=str, default=None,
                        help="Archivo de salida .txt (legacy)")
    
    args = parser.parse_args()
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # V3: Crear StageContext
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    ctx = None
    if HAS_STAGE_UTILS:
        # Inferir experiment si no se proporciona
        if not args.experiment:
            args.experiment = infer_experiment(args)
        
        ctx = StageContext.from_args(
            args,
            stage_number="05",
            stage_slug="analyze_bulk_equations"
        )
        print(f"[V3] Experiment: {ctx.experiment}")
        print(f"[V3] Stage dir: {ctx.stage_dir}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # RESOLVER INPUT
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    input_path = None
    
    # Prioridad 1: --input explicito
    if args.input:
        input_path = Path(args.input).resolve()
    
    # Prioridad 2: V3 - buscar en 03_discover_bulk_equations
    if input_path is None and ctx:
        candidate = ctx.run_root / "03_discover_bulk_equations" / "einstein_discovery_summary.json"
        if candidate.exists():
            input_path = candidate
            print(f"[V3] Input desde stage 03: {input_path}")
    
    # Prioridad 3: --run-dir legacy
    if input_path is None and args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir).resolve()
        bulk_eq_dir = resolve_bulk_equations_dir(run_dir=run_dir)
        if bulk_eq_dir:
            candidate = bulk_eq_dir / "einstein_discovery_summary.json"
            if candidate.exists():
                input_path = candidate
    
    # Prioridad 4: Default legacy
    if input_path is None:
        input_path = Path("sweep_2d_einstein/einstein_discovery_summary.json").resolve()
    
    if not input_path.exists():
        print(f"[ERROR] No existe input: {input_path}")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "input_not_found"})
        return 2
    
    print(f"Analizando: {input_path}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # ANÃLISIS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    try:
        results = load_and_analyze(input_path)
        patterns = find_universal_structure(results)
        report = generate_report(results, patterns)
    except Exception as e:
        print(f"[ERROR] Fallo en analisis: {e}")
        if ctx:
            ctx.write_summary(status="ERROR", counts={"error": str(e)})
        return 3
    
    print(report)
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # RESOLVER OUTPUT
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if args.output:
        output_path = Path(args.output).resolve()
    elif ctx:
        output_path = ctx.stage_dir / "bulk_equations_report.txt"
    elif args.run_dir:
        output_dir = Path(args.run_dir).resolve() / "bulk_equations_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "bulk_equations_report.txt"
    else:
        output_path = Path("equation_analysis.txt").resolve()
    
    # Crear directorio si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar reporte TXT
    output_path.write_text(report)
    print(f"\n-> Guardado: {output_path}")
    
    # Guardar JSON estructurado
    json_output = {
        "by_family": {k: v for k, v in results["by_family"].items()},
        "patterns": {
            "universal_terms": list(patterns["universal_terms"]),
            "family_specific_terms": {k: list(v) for k, v in patterns["family_specific_terms"].items()},
            "coefficient_trends": patterns["coefficient_trends"]
        },
        "stats": {
            "n_geometries": len(results["by_geometry"]),
            "avg_r2": float(np.mean([g["R_r2"] for g in results["by_geometry"].values()])) if results["by_geometry"] else 0.0
        }
    }
    
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(json_output, indent=2, default=str))
    print(f"-> Guardado: {json_path}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # V3: Registrar artefactos y escribir summary
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if ctx:
        ctx.record_artifact("bulk_equations_report_txt", output_path)
        ctx.record_artifact("bulk_equations_report_json", json_path)
        ctx.record_artifact("input_file", input_path)
        
        ctx.write_summary(
            status="OK",
            counts={
                "geometries_analyzed": len(results["by_geometry"]),
                "families": len(results["by_family"]),
                "avg_r2": json_output["stats"]["avg_r2"]
            }
        )
        ctx.write_manifest()
        print(f"[V3] stage_summary.json escrito")
    
    # Legacy: actualizar run_manifest si corresponde
    elif args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir).resolve()
            update_run_manifest(
                run_dir,
                {
                    "bulk_equations_analysis_dir": str(output_path.parent.relative_to(run_dir)
                                                       if output_path.parent.is_relative_to(run_dir)
                                                       else output_path.parent),
                    "bulk_equations_report": str(json_path.relative_to(run_dir)
                                                 if json_path.is_relative_to(run_dir)
                                                 else json_path),
                }
            )
            print(f"-> Manifest actualizado (legacy)")
        except Exception as e:
            print(f"[WARN] No se pudo actualizar manifest: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
