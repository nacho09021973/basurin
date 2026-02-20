#!/usr/bin/env python3
# 08_build_holographic_dictionary.py
# CUERDAS — Bloque C: Atlas holográfico (diccionario geométrico)
#
# OBJETIVO
#   Construir un atlas holográfico interno a partir de la información de geometría
#   y operadores, organizado por sistema/familia/dimensión:
#     - Listar operadores relevantes por sistema (nombres, Δ, etiquetas).
#     - Agregar metadatos de geometría y clasificación (ads, lifshitz, hvlf, ...).
#
# ENTRADAS
#   - runs/<experiment>/02_emergent_geometry_engine/geometry_emergent/*.h5
#
# SALIDAS (V3)
#   runs/<experiment>/08_build_holographic_dictionary/
#     holographic_dictionary_v3_summary.json
#     stage_summary.json
#
# OPCIONAL: CHECKS DE m²L²
#   - Con flags explícitas, puede calcular m²L² = Δ(Δ-d) como diagnóstico.
#   - IMPORTANTE: estos cálculos son post-hoc y no entran en entrenamiento.
#
# RELACIÓN CON OTROS SCRIPTS
#   - Proporciona el "atlas" interno que se cruza con:
#       * 09_real_data_and_dictionary_contracts.py
#
#
# ============================================================================
# NOTACION DE DIMENSIONES (Sec. 3.14 del texto de referencia)
# ============================================================================
#
#   d   = dimension del boundary CFT (NOT bulk dimension)
#   D   = d + 1 = dimension del bulk AdS
#
#   La relacion masa-dimension (Eq. 3.14 del texto):
#       m^2 L^2 = Delta(Delta - d)
#
#   usa 'd' como dimension del boundary, consistente con la convencion
#   de la correspondencia AdS_{d+1}/CFT_d.
#
#   BOUND DE BREITENLOHNER-FREEDMAN (Eq. 2.42):
#       m^2 R^2 >= -d^2/4
#
#   Modos que violan este bound son inestables (taquionicos no permitidos).
#
# ============================================================================
#
# MIGRADO A V3: 2024-12-23
# PATCH ROUTING_CONTRACT: 2024-12-27

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

# ═══════════════════════════════════════════════════════════════════════════
# V3 INFRASTRUCTURE - PATCH: Probar stage_utils primero, luego tools.stage_utils
# ═══════════════════════════════════════════════════════════════════════════
HAS_STAGE_UTILS = False
StageContext = None
add_standard_arguments = None
infer_experiment = None

# Intentar import desde raíz primero (nuevo estándar)
try:
    from stage_utils import StageContext, add_standard_arguments, infer_experiment
    HAS_STAGE_UTILS = True
except ImportError:
    pass

# Fallback a tools/ (legacy)
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
    from cuerdas_io import resolve_geometry_emergent_dir, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


# =============================================================================
# ROUTING CONTRACT: Validación y resolución de rutas
# =============================================================================

def validate_routing_args(args) -> Tuple[bool, str]:
    """
    Valida que no haya conflictos entre --experiment y --output-summary/--data-dir.
    Según ROUTING_CONTRACT: --experiment es la fuente de verdad.
    """
    has_experiment = getattr(args, 'experiment', None) is not None
    has_output_summary = getattr(args, 'output_summary', None) is not None
    has_data_dir = getattr(args, 'data_dir', None) is not None
    
    if has_experiment and (has_output_summary or has_data_dir):
        return False, (
            "CONFLICTO: --experiment y --output-summary/--data-dir son mutuamente excluyentes.\n"
            "  --experiment es la fuente de verdad (ROUTING_CONTRACT).\n"
            "  --output-summary y --data-dir están DEPRECATED.\n"
            "  Usa solo --experiment."
        )
    
    if has_output_summary:
        warnings.warn(
            "--output-summary está DEPRECATED. Usa --experiment en su lugar.",
            DeprecationWarning,
            stacklevel=2
        )
    
    if has_data_dir:
        warnings.warn(
            "--data-dir está DEPRECATED. Usa --experiment en su lugar.",
            DeprecationWarning,
            stacklevel=2
        )
    
    return True, ""


def resolve_geometry_dir(args, ctx) -> Optional[Path]:
    """
    Resuelve el directorio de geometrías según ROUTING_CONTRACT.
    
    Prioridad:
    1. --data-dir explícito (DEPRECATED)
    2. --experiment → buscar en stage 02
    3. --run-dir legacy con cuerdas_io
    """
    # Prioridad 1: --data-dir explícito
    if args.data_dir:
        geometry_dir = Path(args.data_dir).resolve()
        if geometry_dir.exists():
            warnings.warn(
                "--data-dir está DEPRECATED. Usa --experiment en su lugar.",
                DeprecationWarning
            )
            return geometry_dir
        else:
            print(f"[ERROR] Directorio especificado no existe: {geometry_dir}")
            return None
    
    # Prioridad 2: V3 - buscar en stage 02
    if ctx:
        candidates = [
            ctx.run_root / "02_emergent_geometry_engine" / "geometry_emergent",
            ctx.run_root / "geometry_emergent",  # alias legacy
            ctx.run_root / "01_generate_sandbox_geometries",  # IO CONTRACT fallback
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                print(f"[V3] Geometry dir desde stage 02: {candidate}")
                return candidate
        
        # No encontrado - dar error claro
        print(f"[ERROR] No se encontró geometry_emergent en:")
        for c in candidates:
            print(f"        - {c}")
        print(f"\n        Ejecuta primero: python 02_emergent_geometry_engine.py --experiment {ctx.experiment}")
        return None
    
    # Prioridad 3: --run-dir legacy con cuerdas_io
    if args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir).resolve()
        try:
            return resolve_geometry_emergent_dir(run_dir=run_dir)
        except Exception as e:
            print(f"[ERROR] No se pudo resolver geometry_dir: {e}")
            return None
    
    return None


def resolve_output_file(args, ctx) -> Optional[Path]:
    """
    Resuelve el archivo de salida según ROUTING_CONTRACT.
    
    Prioridad:
    1. --experiment → ctx.stage_dir (V3)
    2. --output-summary explícito (DEPRECATED)
    3. --run-dir / holographic_dictionary (legacy)
    """
    if ctx:
        return ctx.stage_dir / "holographic_dictionary_v3_summary.json"
    
    if args.output_summary:
        return Path(args.output_summary).resolve()
    
    if args.run_dir:
        out_dir = Path(args.run_dir).resolve() / "holographic_dictionary"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "holographic_dictionary_v3_summary.json"
    
    # Fallback local
    return Path("holographic_dictionary_v3_summary.json").resolve()


def safe_relpath(path: Path, base: Path) -> str:
    """Helper que no explota si path no es relativo a base."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def extract_delta_from_correlator(x, G2):
    """
    Extrae Delta de un correlador de 2 puntos, asumiendo:
        G2(x) ~ 1/x^{2Delta}

    Devuelve un dict con:
        - Delta
        - fit_r2
        - status
    """
    x = np.array(x)
    G2 = np.array(G2)

    mask = (x > 0) & (G2 > 0)
    if mask.sum() < 5:
        return {"status": "insufficient_data"}

    logx = np.log(x[mask])
    logG2 = np.log(G2[mask])

    A = np.vstack([logx, np.ones_like(logx)]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, logG2, rcond=None)
    slope, intercept = coeffs
    Delta = -slope / 2.0

    if residuals.size > 0:
        ss_res = residuals[0]
    else:
        ss_res = np.sum((logG2 - (slope * logx + intercept)) ** 2)
    ss_tot = np.sum((logG2 - np.mean(logG2)) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot

    return {"status": "ok", "Delta": Delta, "fit_r2": r2}


def check_breitenlohner_freedman_bound(m2L2: float, d: int) -> dict:
    """
    Verifica el bound de Breitenlohner-Freedman (Eq. 2.42 del texto).
    
    El bound establece que para modos estables en AdS:
        m^2 R^2 >= -d^2/4
    
    donde d es la dimension del boundary CFT.
    
    Modos que violan este bound son taquionicos inestables y no corresponden
    a operadores unitarios en la CFT dual.
    
    Args:
        m2L2: masa al cuadrado en unidades del radio AdS (m^2 L^2)
        d: dimension del boundary CFT
    
    Returns:
        dict con:
            - bf_bound: valor del bound (-d^2/4)
            - satisfies_bf: True si m2L2 >= bf_bound
            - margin: diferencia m2L2 - bf_bound
            - status: "stable", "marginal", o "unstable"
    """
    bf_bound = -d * d / 4.0
    margin = m2L2 - bf_bound
    satisfies = m2L2 >= bf_bound
    
    if margin > 0.1:
        status = "stable"
    elif margin >= 0:
        status = "marginal"
    else:
        status = "unstable"
    
    return {
        "bf_bound": bf_bound,
        "satisfies_bf": satisfies,
        "margin": margin,
        "status": status,
    }


def discover_mass_dimension_relation(Deltas, m2L2, d, seed=42):
    """
    Usa PySR para descubrir la relacion entre Delta y m²L².
    Devuelve un dict con:
        - discovered_equation (str)
        - r2 (float)
        - status
        - holographic_r2 (ajuste si forzamos m²L² = Delta(Delta-d))
    """
    if not HAS_PYSR:
        return {"status": "pysr_not_available"}
    
    results = {}
    Deltas = np.array(Deltas).reshape(-1, 1)
    m2L2 = np.array(m2L2).reshape(-1, 1)

    X = np.hstack([Deltas, np.full_like(Deltas, d)])
    y = m2L2

    if len(X) < 5:
        results["status"] = "insufficient_data"
        return results

    model = PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        elementwise_loss="L2DistLoss()",
        maxsize=12,
        model_selection="best",
        progress=False,
        verbosity=0,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
    )
    model.fit(X, y)
    best = model.get_best()
    y_pred = model.predict(X)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    results["discovered_equation"] = str(best["equation"])
    results["r2"] = float(r2)
    results["status"] = "ok"

    # Comparacion con Delta(Delta-d) (chequeo teórico, no label)
    Deltas_flat = Deltas.reshape(-1)
    m2L2_flat = m2L2.reshape(-1)
    valid = ~np.isnan(Deltas_flat) & ~np.isnan(m2L2_flat)
    if valid.sum() > 3:
        Deltas_valid = Deltas_flat[valid]
        m2L2_valid = m2L2_flat[valid]
        holo_pred = Deltas_valid * (Deltas_valid - d)
        ss_res_holo = np.sum((m2L2_valid - holo_pred) ** 2)
        ss_tot_holo = np.sum((m2L2_valid - np.mean(m2L2_valid)) ** 2) + 1e-10
        results["holographic_r2"] = float(1 - ss_res_holo / ss_tot_holo)
        
        # === VERIFICACION BREITENLOHNER-FREEDMAN (POST-HOC) ===
        # Eq. 2.42: m^2 R^2 >= -d^2/4
        bf_checks = [check_breitenlohner_freedman_bound(m2, d) for m2 in m2L2_valid]
        n_stable = sum(1 for c in bf_checks if c["status"] == "stable")
        n_marginal = sum(1 for c in bf_checks if c["status"] == "marginal")
        n_unstable = sum(1 for c in bf_checks if c["status"] == "unstable")
        results["bf_bound_check"] = {
            "d": d,
            "bf_bound_value": -d * d / 4.0,
            "n_modes_checked": len(bf_checks),
            "n_stable": n_stable,
            "n_marginal": n_marginal,
            "n_unstable": n_unstable,
            "all_satisfy_bf": n_unstable == 0,
        }
    else:
        results["holographic_r2"] = None
        results["bf_bound_check"] = None

    return results


# =============================================================================
# ARGPARSE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase XI: construir diccionario holográfico agrupando por (family, d)"
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # V3: Argumentos estándar
    # ═══════════════════════════════════════════════════════════════════════
    if HAS_STAGE_UTILS and add_standard_arguments:
        add_standard_arguments(parser)
    else:
        parser.add_argument("--experiment", type=str, default=None,
                            help="Nombre del experimento (fuente de verdad)")
        parser.add_argument("--run-dir", type=str, default=None,
                            help="[DEPRECATED] Usar --experiment")
    
    # Argumentos legacy (compatibilidad - DEPRECATED)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="[DEPRECATED] Directorio con los .h5 de geometría. Usar --experiment.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="[DEPRECATED] Fichero JSON de salida. Usar --experiment.",
    )
    
    # Argumentos específicos del script
    parser.add_argument(
        "--mass-source",
        type=str,
        default="hdf5",
        choices=["hdf5", "emergent"],
        help="Fuente de masas: 'hdf5' (ground truth/control) o 'emergent' (extraer solo Delta de correladores)",
    )
    parser.add_argument(
        "--compute-m2-from-delta",
        action="store_true",
        help="(MODO CONTROL, solo con mass_source=hdf5) Si no hay m2L2 en HDF5, calcula m²L² = Delta(Delta-d)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para PySR",
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ROUTING CONTRACT: Validar conflictos
    # ═══════════════════════════════════════════════════════════════════════
    is_valid, error_msg = validate_routing_args(args)
    if not is_valid:
        print(f"[ERROR] {error_msg}")
        return 1
    
    # ═══════════════════════════════════════════════════════════════════════
    # V3: Crear StageContext
    # ═══════════════════════════════════════════════════════════════════════
    ctx = None
    if HAS_STAGE_UTILS and StageContext:
        if not getattr(args, 'experiment', None):
            if infer_experiment:
                args.experiment = infer_experiment(args)
        
        if args.experiment:
            ctx = StageContext.from_args(
                args,
                stage_number="08",
                stage_slug="build_holographic_dictionary"
            )
            print(f"[V3] Experiment: {ctx.experiment}")
            print(f"[V3] Stage dir: {ctx.stage_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESOLVER INPUT (geometry_dir)
    # ═══════════════════════════════════════════════════════════════════════
    geometry_dir = resolve_geometry_dir(args, ctx)
    
    if geometry_dir is None:
        print("[ERROR] Debe proporcionar --experiment, --run-dir o --data-dir")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "no_geometry_dir"})
        return 2
    
    if not geometry_dir.exists() or not geometry_dir.is_dir():
        print(f"[ERROR] geometry_dir no es un directorio válido: {geometry_dir}")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "geometry_dir_not_found"})
        return 2
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESOLVER OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    output_file = resolve_output_file(args, ctx)

    print("=" * 70)
    print("FASE XI - DICCIONARIO HOLOGRÁFICO v3.1 (FIX EMERGENT)")
    print("=" * 70)
    print(f"Mass source: {args.mass_source.upper()}")
    if args.mass_source == "hdf5":
        print("   MODO CONTROL: Usando ground-truth de HDF5 (Delta_mass_dict)")
        if args.compute_m2_from_delta:
            print("   -> [CONTROL] Si falta m2L2, se calcula m²L² = Delta(Delta-d)")
    else:
        print("   MODO EMERGENTE: Extrayendo Delta de correladores")
        print("   -> NO se calcula m²L² en esta fase (solo atlas de Δ)")
    print("=" * 70)

    data_by_family_d = defaultdict(
        lambda: {
            "family": None,
            "d": None,
            "Deltas": [],
            "m2L2": [],
            "geometries": [],
            "operators": [],
        }
    )

    geometry_results = []

    # Recorremos todos los .h5 de la carpeta de geometría
    h5_files = sorted(geometry_dir.glob("*.h5"))
    if not h5_files:
        print(f"[WARN] No se encontraron .h5 en {geometry_dir}")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "no_h5_files"})
        return 2

    for h5_path in h5_files:
        name = h5_path.stem

        with h5py.File(h5_path, "r") as f:
            family = f.attrs.get("family", "unknown")
            if isinstance(family, bytes):
                family = family.decode("utf-8")
            # NOTA: 'd' es la dimension del BOUNDARY CFT, no del bulk
            # El bulk tiene dimension D = d + 1 (convencion AdS_{d+1}/CFT_d)
            try:
                d = int(f.attrs.get("d", 4))
            except (TypeError, ValueError):
                d = 4

            key = f"{family}_d{d}"
            data_by_family_d[key]["family"] = family
            data_by_family_d[key]["d"] = d

            geo_result = {
                "name": name,
                "family": family,
                "d": d,
                "operators_extracted": [],
            }

            # === MODO HDF5 (control) ===
            if args.mass_source == "hdf5":
                Delta_mass_dict = {}
                if "boundary" in f and "Delta_mass_dict" in f["boundary"].attrs:
                    try:
                        raw = f["boundary"].attrs["Delta_mass_dict"]
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        Delta_mass_dict = json.loads(raw)
                    except Exception as e:
                        print(f"   {name}: error leyendo Delta_mass_dict: {e}")

                for op_name, info in Delta_mass_dict.items():
                    if isinstance(info, dict):
                        Delta = info.get("Delta")
                        m2L2 = info.get("m2L2")
                    else:
                        Delta = info
                        m2L2 = None

                    if Delta is None:
                        continue

                    # Si falta m2L2 y se pide calcular
                    if m2L2 is None and args.compute_m2_from_delta:
                        m2L2 = Delta * (Delta - d)
                        m2L2_method = "computed_from_delta"
                    elif m2L2 is not None:
                        m2L2_method = "hdf5_ground_truth"
                    else:
                        m2L2_method = "not_available"

                    print(
                        f"   {name}/{op_name}: Δ={Delta:.3f}, m²L²={m2L2 if m2L2 else 'N/A'} "
                        f"[{m2L2_method}]"
                    )

                    geo_result["operators_extracted"].append(
                        {
                            "name": op_name,
                            "Delta": Delta,
                            "m2L2": m2L2,
                            "m2L2_method": m2L2_method,
                        }
                    )

                    data_by_family_d[key]["Deltas"].append(Delta)
                    if m2L2 is not None:
                        data_by_family_d[key]["m2L2"].append(m2L2)
                    data_by_family_d[key]["operators"].append(
                        {
                            "name": f"{name}_{op_name}",
                            "Delta": Delta,
                            "m2L2": m2L2,
                            "source_geometry": name,
                            "m2L2_method": m2L2_method,
                        }
                    )

            # === MODO EMERGENT ===
            else:
                boundary = f.get("boundary", {})
                x_grid = boundary.get("x_grid", None)
                if x_grid is None:
                    print(f"   {name}: sin x_grid, skip")
                    continue
                x_grid = x_grid[:]

                operators_attr = boundary.attrs.get("operators", "[]")
                if isinstance(operators_attr, bytes):
                    operators_attr = operators_attr.decode("utf-8")
                try:
                    operators = json.loads(operators_attr)
                except json.JSONDecodeError:
                    operators = []

                for op in operators:
                    op_name = op.get("name")
                    if op_name is None:
                        continue

                    G2_key = f"G2_{op_name}"
                    if G2_key not in boundary:
                        continue

                    G2 = boundary[G2_key][:]
                    result = extract_delta_from_correlator(x_grid, G2)
                    if result["status"] != "ok":
                        continue

                    Delta = result["Delta"]

                    print(
                        f"   {name}/{op_name}: Delta={Delta:.3f} "
                        f"[emergent, m²L² NO calculado en esta fase]"
                    )

                    geo_result["operators_extracted"].append(
                        {
                            "name": op_name,
                            "Delta": Delta,
                            "Delta_fit_r2": result.get("fit_r2"),
                        }
                    )

                    data_by_family_d[key]["Deltas"].append(Delta)
                    data_by_family_d[key]["operators"].append(
                        {
                            "name": f"{name}_{op_name}",
                            "Delta": Delta,
                            "source_geometry": name,
                            "m2L2_method": "not_available",
                        }
                    )

            data_by_family_d[key]["geometries"].append(name)
            geometry_results.append(geo_result)

    # Construir by_system solo cuando tenemos m2L2 disponible
    by_system = {}
    for key, fdata in sorted(data_by_family_d.items()):
        n = min(len(fdata["Deltas"]), len(fdata["m2L2"]))
        if n == 0:
            print(f"   {key}: 0 puntos con m2L2 disponible [SKIP]")
            continue

        Deltas = fdata["Deltas"][:n]
        m2L2_list = fdata["m2L2"][:n]

        by_system[key] = {
            "family": fdata["family"],
            "d": fdata["d"],
            "n_points": n,
            "Delta": Deltas,
            "m2L2_emergent": m2L2_list,
            "geometries_included": fdata["geometries"],
            "source": args.mass_source,
        }
        print(
            f"   {key}: {n} puntos con m2L2, "
            f"d={fdata['d']}, {len(fdata['geometries'])} geometrías"
        )

    # Descubrir relaciones masa-dimension donde sea posible
    discovery_results = {}
    for key, sdata in by_system.items():
        if sdata["n_points"] < 3:
            print(f"   {key}: datos insuficientes ({sdata['n_points']} < 3)")
            discovery_results[key] = {"status": "insufficient_data"}
            continue

        print(f"\n>> Descubriendo para '{key}' (d={sdata['d']})...")
        result = discover_mass_dimension_relation(
            np.array(sdata["Delta"]),
            np.array(sdata["m2L2_emergent"]),
            sdata["d"],
            seed=args.seed,
        )
        discovery_results[key] = result
        if result["status"] == "ok":
            print(f"   Mejor ecuación: {result['discovered_equation']}")
            print(f"   R²(PySR): {result['r2']:.4f}")
            if result.get("holographic_r2") is not None:
                print(f"   R²(m²L²=Delta(Delta-d)): {result['holographic_r2']:.4f}")
        else:
            print(f"   Status: {result['status']}")

    summary = {
        "by_system": by_system,
        "discoveries": discovery_results,
        "geometry_results": geometry_results,
        "mass_source": args.mass_source,
        "compute_m2_from_delta": args.compute_m2_from_delta,
        "version": "v3.1_routing_contract",
        "notes": [
            "v3.1: FIX MODO EMERGENT (no mezcla d ni masas entre geometrías)",
            "rev.honestidad: en modo emergent no se calcula m²L² en este script; "
            "las masas deben venir de datos externos (HDF5 u otros módulos).",
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))
    print(f"\nResumen guardado en: {output_file}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # V3: Registrar artefactos y escribir summary
    # ═══════════════════════════════════════════════════════════════════════
    if ctx:
        ctx.record_artifact("holographic_dictionary_summary", output_file)
        ctx.record_artifact("geometry_dir_input", geometry_dir)
        
        ctx.write_summary(
            status="OK" if len(by_system) > 0 else "WARNING",
            counts={
                "h5_files_scanned": len(h5_files),
                "systems_with_m2L2": len(by_system),
                "geometries_processed": len(geometry_results),
                "discoveries_made": len([d for d in discovery_results.values() if d.get("status") == "ok"]),
            }
        )
        ctx.write_manifest()
        print(f"[V3] stage_summary.json escrito")
    
    # Legacy: actualizar run_manifest
    elif args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir).resolve()
            update_run_manifest(
                run_dir,
                {
                    "holographic_dictionary_dir": safe_relpath(output_file.parent, run_dir),
                    "holographic_dictionary_summary": safe_relpath(output_file, run_dir),
                }
            )
            print(f"Manifest actualizado (legacy)")
        except Exception as e:
            print(f"[WARN] No se pudo actualizar manifest: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
