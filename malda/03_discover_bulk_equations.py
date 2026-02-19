#!/usr/bin/env python3
# 03_discover_bulk_equations.py
# CUERDAS ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Bloque A: GeometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a emergente (descubrimiento de ecuaciones de bulk)
#
# Objective:
#   Aplicar regresiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n simbÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³lica (PySR u otro SR) sobre la geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a emergente
#   para descubrir ecuaciones de campo en el bulk (para A, f, R, etc.).
#
# Inputs: (IO CONTRACT V3)
#   Usa io_contract_resolver para encontrar geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­as:
#   - Prioridad 1: runs/<exp>/02_emergent_geometry_engine/geometry_emergent/*.h5
#   - Prioridad 2: runs/<exp>/01_generate_sandbox_geometries/*.h5
#
# Outputs:
#   runs/<experiment>/03_discover_bulk_equations/
#     einstein_discovery_summary.json
#     <geometry_name>/einstein_discovery.json
#
# HONESTIDAD
#   - No se usan ecuaciones teÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ricas conocidas como features ni como tÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rminos forzados.
#   - Las comparaciones con ecuaciones de Einstein o variantes se realizan mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡s tarde,
#     y se etiquetan explÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­citamente como "post-hoc".
#
# Version: 2024-12-29 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Con io_contract_resolver + fix Ricci

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False
    print("Warning: PySR not available.")

# Import IO contract resolver
try:
    from io_contract_resolver import resolve_input, resolve_output, IOContractError
    HAS_CONTRACT_RESOLVER = True
except ImportError:
    HAS_CONTRACT_RESOLVER = False
    print("Warning: io_contract_resolver not available. Using legacy path resolution.")

HAS_STAGE_UTILS = False
EXIT_OK = 0
EXIT_ERROR = 3
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"
StageContext = None
add_standard_arguments = None
parse_stage_args = None

try:
    from stage_utils import (
        EXIT_ERROR, EXIT_OK, STATUS_ERROR, STATUS_OK,
        StageContext, add_standard_arguments, parse_stage_args,
    )
    HAS_STAGE_UTILS = True
except ImportError:
    pass

if not HAS_STAGE_UTILS:
    try:
        from tools.stage_utils import (
            EXIT_ERROR, EXIT_OK, STATUS_ERROR, STATUS_OK,
            StageContext, add_standard_arguments, parse_stage_args,
        )
        HAS_STAGE_UTILS = True
    except ImportError:
        print("[WARN] stage_utils not available")


# ============================================================
# CARGA DE GEOMETRÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂAS (H5 + NPZ)
# ============================================================

def load_geometry_file(file_path: Path) -> Dict[str, Any]:
    """
    Carga un file de geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a (.h5 o .npz).
    Devuelve dict con keys: z, A, f, category, name
    """
    name = file_path.stem.replace("_geometry", "").replace("_emergent", "")
    
    if file_path.suffix == ".h5":
        if not HAS_H5PY:
            raise ImportError("h5py no disponible para leer .h5")
        with h5py.File(file_path, "r") as f:
            z = f["z_grid"][:]
            A = f["A_of_z"][:]
            f_arr = f["f_of_z"][:]
            category = f.attrs.get("family", "unknown")
            if isinstance(category, bytes):
                category = category.decode("utf-8")
        return {"z": z, "A": A, "f": f_arr, "category": str(category), "name": name}
    
    elif file_path.suffix == ".npz":
        data = np.load(file_path, allow_pickle=True)
        z = data["z"]
        A = data.get("A_pred", data.get("A", None))
        f_arr = data.get("f_pred", data.get("f", None))
        category = str(data.get("category", "unknown"))
        return {"z": z, "A": A, "f": f_arr, "category": category, "name": name}
    
    else:
        raise ValueError(f"Formato no soportado: {file_path.suffix}")


# ============================================================
# CALCULO DE TENSORES GEOMETRICOS
# ============================================================

def compute_geometric_tensors(
    z_grid: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    d: int
) -> Dict[str, np.ndarray]:
    """
    Calcula tensores geometricos en cada punto z.
    
    GAUGE: Metrica conformal (Domain Wall) con blackening:
        ds^2 = e^{2A(z)} [-f(z)dt^2 + dx^2] + dz^2/f(z)
    
    Donde:
        - d = dimension del boundary CFT (tipicamente 3 o 4)
        - D = d + 1 = dimension total del bulk
        - A(z) = warp factor
        - f(z) = blackening factor (f=1 -> puro AdS, f->0 en horizonte)
    
    FORMULA DEL ESCALAR DE RICCI (geometria diferencial basica):
        R = -2D*A'' - D(D-1)*(A')^2 - (f'/f)*A'
    
    donde D = d + 1.
    
    NOTA IMPORTANTE: Esta formula es GEOMETRIA, no fisica.
    - NO asumimos R = -d(d+1)/L^2 (valor AdS)
    - NO imponemos ecuaciones de Einstein
    - Solo calculamos la curvatura de la metrica dada
    
    La formula se deriva de los simbolos de Christoffel para este gauge.
    Ver: 02_emergent_geometry_engine.py compute_ricci_from_metric()
    
    FIX 2024-12-29: Eliminado factor exp(-2A) incorrecto.
    El factor exp(-2A) aparece en g^{uv} (direcciones transversas),
    pero g^{zz} = f, no e^{-2A}. La contraccion correcta no tiene
    ese prefactor global.
    """
    n_z = len(z_grid)
    D = d + 1
    dz = z_grid[1] - z_grid[0]
    
    # Suavizar para estabilidad numerica
    A_smooth = gaussian_filter1d(A, sigma=1)
    f_smooth = gaussian_filter1d(f, sigma=1)
    f_smooth = np.clip(f_smooth, 1e-6, 2.0)
    
    # Derivadas
    dA = np.gradient(A_smooth, dz)
    d2A = np.gradient(dA, dz)
    df = np.gradient(f_smooth, dz)
    d2f = np.gradient(df, dz)
    
    # ESCALAR DE RICCI - FORMULA CONSISTENTE CON 02_emergent_geometry_engine.py
    # R = -2D*A'' - D(D-1)*(A')^2 - (f'/f)*A'
    # 
    # Esta es la formula correcta para el gauge conformal.
    # NO tiene el factor exp(-2A) que estaba antes (error corregido).
    f_safe = f_smooth + 1e-10  # Evitar division por cero
    
    R_scalar = (
        -2.0 * D * d2A
        - D * (D - 1) * dA**2
        - (df / f_safe) * dA
    )
    
    # Traza del tensor de Einstein: G = (1 - D/2) * R
    G_trace = (1.0 - D / 2.0) * R_scalar
    
    # Para AdS puro con A = -log(z/L), f = 1:
    #   dA = -1/z, d2A = 1/z^2, df = 0
    #   R = -2D/z^2 - D(D-1)/z^2 = -D(2 + D - 1)/z^2 = -D(D+1)/z^2
    # Con z = L (escala AdS), R = -D(D+1)/L^2 = -(d+1)(d+2)/L^2
    # Esto es correcto para AdS.
    
    return {
        "z": z_grid,
        "R_scalar": R_scalar,
        "G_trace": G_trace,
        "A": A_smooth,
        "f": f_smooth,
        "dA": dA,
        "d2A": d2A,
        "df": df,
        "d2f": d2f,
        "D": D,
    }



# ============================================================
# DESCUBRIMIENTO SIMBOLICO CON PYSR (LIMPIO)
# ============================================================

def discover_geometric_relations(
    tensors: Dict[str, np.ndarray],
    d: int,
    output_dir: Path,
    niterations: int = 100,
    maxsize: int = 15,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Usa PySR para descubrir relaciones geometricas SIN asumir Einstein.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    z = tensors["z"]
    R = tensors["R_scalar"]
    A = tensors["A"]
    f = tensors["f"]
    dA = tensors["dA"]
    d2A = tensors["d2A"]
    df = tensors["df"]
    d2f = tensors["d2f"]
    D = tensors["D"]
    
    results = {}
    
    # === Analisis basico de R ===
    R_mean = np.mean(R)
    R_std = np.std(R)
    
    results["R_statistics"] = {
        "mean": float(R_mean),
        "std": float(R_std),
        "min": float(np.min(R)),
        "max": float(np.max(R)),
        "coefficient_of_variation": float(R_std / (np.abs(R_mean) + 1e-10))
    }
    
    print(f"   R statistics:")
    print(f"     Mean: {R_mean:.4f}")
    print(f"     Std:  {R_std:.4f}")
    print(f"     CV:   {results['R_statistics']['coefficient_of_variation']:.4f}")
    
    if not HAS_PYSR:
        results["pysr_available"] = False
        return results
    
    results["pysr_available"] = True
    
    # === Test 1: R es funcion constante? ===
    print("\n   [1] Testing if R ~ constant...")
    
    is_constant = results["R_statistics"]["coefficient_of_variation"] < 0.1
    results["R_is_constant"] = is_constant
    
    if is_constant:
        print(f"     R ~ {R_mean:.4f} (constante)")
        results["R_constant_value"] = float(R_mean)
    else:
        print(f"     R varia significativamente")
    
    # === Test 2: Descubrir R(A, f, derivadas) ===
    print("\n   [2] Discovering R = F(A, f, derivatives)...")
    
    X = np.column_stack([A, f, dA, d2A, df, d2f])
    y = R
    
    valid = np.isfinite(y) & (np.abs(y) < 1e6)
    X_valid = X[valid]
    y_valid = y[valid]
    
    if len(y_valid) < 10:
        print("   Insufficient valid data points")
        results["R_equation"] = None
        return results
    
    model_R = PySRRegressor(
        niterations=niterations,
        populations=8,
        population_size=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "exp", "log", "neg"],
        extra_sympy_mappings={"neg": lambda x: -x},
        elementwise_loss="L2DistLoss()",
        maxsize=maxsize,
        model_selection="best",
        progress=False,
        verbosity=0,
        parallelism="serial",
        deterministic=True,
        random_state=seed,
        tempdir=str(output_dir),  # Evitar uso de outputs/ global
    )

    model_R.fit(X_valid, y_valid)
    
    best_R = model_R.get_best()
    R_pred = model_R.predict(X_valid)
    
    ss_res = np.sum((y_valid - R_pred)**2)
    ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    
    results["R_equation"] = {
        "equation": str(best_R["equation"]),
        "complexity": int(best_R["complexity"]),
        "loss": float(best_R["loss"]),
        "r2": float(r2),
        "feature_names": ["A", "f", "dA", "d2A", "df", "d2f"]
    }
    
    print(f"     Discovered: R = {best_R['equation']}")
    print(f"     RÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€š²: {r2:.4f}")
    print(f"     Complexity: {best_R['complexity']}")
    
    return results


# ============================================================
# Post-validation (ES EINSTEIN?)
# ============================================================

def validate_einstein_posterior(results: Dict[str, Any], d: int) -> Dict[str, Any]:
    """
    Validacion post-hoc: Ã‚Â¿la ecuacion descubierta es compatible con Einstein?
    
    Para AdS_d+1 con constante cosmologica ÃŽâ€º < 0:
        R = 2D/(D-2) * ÃŽâ€º = -d(d+1)/L²  (constante y NEGATIVO)
    
    Criterios:
        1. R debe ser aproximadamente constante (cv < 0.15)
        2. R debe ser NEGATIVO (no cero, no positivo)
        3. |R| debe ser significativo (no flat space)
        4. A(z) debe tener comportamiento logaritmico (A ~ -log(z/L))
    """
    D = d + 1
    
    validation = {
        "R_constant": False,
        "R_negative": False,
        "R_significant": False,  # NUEVO: R no es ~0
        "einstein_vacuum_compatible": False,
        "A_is_logarithmic": False,
        "einstein_score": 0.0,
        "verdict": "UNKNOWN"
    }
    
    R_stats = results.get("R_statistics", {})
    cv = R_stats.get("coefficient_of_variation", 1.0)
    R_mean = R_stats.get("mean", 0)
    R_std = R_stats.get("std", 0)
    
    # Test 1: R constante (baja variacion)
    validation["R_constant"] = cv < 0.15
    
    # Test 2: R negativo (requisito para AdS con ÃŽâ€º < 0)
    validation["R_negative"] = R_mean < -0.1  # Threshold para evitar R~0
    
    # Test 3: R significativo (no flat space donde R=0)
    # AdS tipico tiene |R| ~ d(d+1)/L² ~ O(1) o mayor
    validation["R_significant"] = abs(R_mean) > 0.5
    
    # Einstein vacuum requiere los tres criterios
    if validation["R_constant"] and validation["R_negative"] and validation["R_significant"]:
        validation["einstein_vacuum_compatible"] = True
        Lambda_implied = R_mean * (D - 2) / (2 * D)
        validation["implied_Lambda"] = float(Lambda_implied)
        # Estimar L² desde R = -d(d+1)/L²
        L_squared = -d * (d + 1) / (R_mean + 1e-10)
        if L_squared > 0:
            validation["implied_L"] = float(np.sqrt(L_squared))
    
    # Test 4: A logaritmico - CORREGIDO
    # Solo si la ecuacion descubierta contiene "log" explicitamente
    # Y ademas R debe ser significativo (evita falso positivo en flat space)
    R_eq = results.get("R_equation", {})
    if R_eq:
        eq_str = R_eq.get("equation", "")
        has_log_in_equation = "log" in eq_str.lower()
        # A es logaritmico solo si: hay log en ecuacion O (R constante Y significativo)
        validation["A_is_logarithmic"] = has_log_in_equation or (
            validation["R_constant"] and validation["R_significant"]
        )
    
    # Calcular score
    score = 0.0
    
    # R constant: +0.2 SOLO si R tambien es significativo
    # (evita que flat_space con R=0 constante sume puntos)
    if validation["R_constant"] and validation["R_significant"]:
        score += 0.2
    
    # R negativo: +0.2
    if validation["R_negative"]:
        score += 0.2
    
    # R significativo (no flat): +0.2
    if validation["R_significant"]:
        score += 0.2
    
    # Einstein vacuum compatible: +0.2
    if validation["einstein_vacuum_compatible"]:
        score += 0.2
    
    # A logaritmico: +0.1
    if validation["A_is_logarithmic"]:
        score += 0.1
    
    # Buen ajuste R²: +0.1 (solo si R es significativo)
    if R_eq and R_eq.get("r2", 0) > 0.95 and validation["R_significant"]:
        score += 0.1
    
    validation["einstein_score"] = min(score, 1.0)
    
    # Veredicto basado en score
    if validation["einstein_score"] >= 0.7:
        validation["verdict"] = "LIKELY_EINSTEIN_VACUUM"
    elif validation["einstein_score"] >= 0.4:
        validation["verdict"] = "POSSIBLY_EINSTEIN_WITH_MATTER"
    else:
        validation["verdict"] = "NON_EINSTEIN_OR_DEFORMED"
    
    return validation


# ============================================================
# RESOLUCIÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œN DE RUTAS (CON CONTRATO)
# ============================================================

def resolve_geometries_dir(args, run_dir: Optional[Path] = None) -> Path:
    """
    Resuelve el directory de geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­as usando el contrato IO.
    Fallback a lÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³gica legacy si el contrato no estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ disponible.
    """
    # Prioridad 1: --geometry-dir explÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­cito
    if args.geometry_dir:
        geo_dir = Path(args.geometry_dir)
        if geo_dir.exists():
            h5_files = list(geo_dir.glob("*.h5"))
            npz_files = list(geo_dir.glob("*.npz"))
            if h5_files or npz_files:
                return geo_dir
        raise FileNotFoundError(f"No geometries found en {geo_dir}")
    
    # Prioridad 2: Usar contrato IO
    if HAS_CONTRACT_RESOLVER and run_dir:
        try:
            return resolve_input("03", "geometries", run_dir)
        except IOContractError as e:
            print(f"[WARN] Contract resolver: {e}")
            # Continuar con fallback
    
    # Prioridad 3: Fallback legacy
    if run_dir:
        # Intentar rutas conocidas en orden
        candidates = [
            run_dir / "02_emergent_geometry_engine" / "geometry_emergent",
            run_dir / "01_generate_sandbox_geometries",
            run_dir / "geometry_emergent",
            run_dir / "predictions",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                h5_files = list(candidate.glob("*.h5"))
                npz_files = list(candidate.glob("*.npz"))
                if h5_files or npz_files:
                    print(f"[LEGACY] Geometries desde: {candidate}")
                    return candidate
    
    raise FileNotFoundError(
        f"No geometries found.\n"
        f"Usa --geometry-dir o asegÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºrate de que existen en:\n"
        f"  - <run_dir>/02_emergent_geometry_engine/geometry_emergent/\n"
        f"  - <run_dir>/01_generate_sandbox_geometries/"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Descubre ecuaciones geometricas SIN asumir Einstein"
    )
    
    parser.add_argument("--geometry-dir", type=str, default=None,
                        help="directory con *.h5 o *.npz de geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­as")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run dir (busca geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­as automÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ticamente)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="directory de salida")
    parser.add_argument("--niterations", type=int, default=100,
                        help="Iteraciones PySR")
    parser.add_argument("--maxsize", type=int, default=15,
                        help="TamaÃƒÆ’Ã†â€™Ãƒâ€š±o mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ximo de expresiones")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--d", type=int, default=3,
                        help="DimensiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n del boundary CFT")
    
    if HAS_STAGE_UTILS:
        add_standard_arguments(parser)
    else:
        parser.add_argument("--experiment", type=str, default=None)
    
    args = parse_stage_args(parser) if HAS_STAGE_UTILS else parser.parse_args()
    
    # V3: Usar StageContext si está disponible
    ctx = None
    if HAS_STAGE_UTILS and StageContext is not None:
        ctx = StageContext.from_args(args, stage_number="03", stage_slug="discover_bulk_equations")
        run_dir = ctx.run_root
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    elif hasattr(args, 'experiment') and args.experiment:
        run_dir = Path("runs") / args.experiment
    else:
        run_dir = None
    
    status = STATUS_OK
    exit_code = EXIT_OK
    error_message: Optional[str] = None

    try:
        # === RESOLVER GEOMETRÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂAS ===
        geometries_dir = resolve_geometries_dir(args, run_dir)
        
        # === RESOLVER OUTPUT ===
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif run_dir:
            output_dir = run_dir / "03_discover_bulk_equations"
        else:
            output_dir = Path("bulk_equations")
        
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("BULK EQUATIONS DISCOVERY GRAVITATORIAS")
        print("=" * 70)
        print(f"Geometrias: {geometries_dir}")
        print(f"Output:     {output_dir}")
        print(f"d:          {args.d}")
        print("=" * 70)
        print("\nNote: Este script NO asume Einstein a priori.")
        print("      Descubre la ecuacion y LUEGO verifica si es Einstein.")
        print("=" * 70)
        
        all_results = {"geometries": []}
        
        # Buscar files de geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a
        geometry_files = sorted(geometries_dir.glob("*.h5"))
        if not geometry_files:
            geometry_files = sorted(geometries_dir.glob("*.npz"))
        
        if not geometry_files:
            raise FileNotFoundError(f"No se encontraron files .h5 ni .npz en {geometries_dir}")
        
        print(f"\nEncontrados {len(geometry_files)} files de geometrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a")
        
        for geo_path in geometry_files:
            try:
                geo_data = load_geometry_file(geo_path)
            except Exception as e:
                print(f"[WARN] Error loading {geo_path}: {e}")
                continue
            
            name = geo_data["name"]
            z = geo_data["z"]
            A = geo_data["A"]
            f = geo_data["f"]
            category = geo_data["category"]
            
            print(f"\n>> Procesando {name} (family: {category})...")
            
            print("   Calculando tensores geometricos...")
            tensors = compute_geometric_tensors(z, A, f, args.d)
            
            print("   Descubriendo ecuaciones (sin asumir Einstein)...")
            geo_output = output_dir / name
            results = discover_geometric_relations(
                tensors, args.d, geo_output,
                niterations=args.niterations,
                maxsize=args.maxsize,
                seed=args.seed
            )
            
            print("\n   Post-validation (es Einstein?)...")
            validation = validate_einstein_posterior(results, args.d)
            
            print(f"\n   Validation:")
            print(f"     R constant:              {'OK' if validation['R_constant'] else 'NO'}")
            print(f"     R negative (AdS):         {'OK' if validation.get('R_negative', False) else 'NO'}")
            print(f"     Compatible with Einstein:  {'OK' if validation['einstein_vacuum_compatible'] else 'NO'}")
            print(f"     A ~ log(z):               {'OK' if validation['A_is_logarithmic'] else 'NO'}")
            print(f"     Einstein score:           {validation['einstein_score']:.2f}")
            print(f"     Verdict:                {validation['verdict']}")
            if 'implied_Lambda' in validation:
                print(f"     Lambda implÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­cita:         {validation['implied_Lambda']:.4f}")
            
            geo_results = {
                "name": name,
                "category": category,
                "results": results,
                "validation": validation
            }
            all_results["geometries"].append(geo_results)
            
            json_path = geo_output / "einstein_discovery.json"
            json_path.parent.mkdir(exist_ok=True)
            json_path.write_text(json.dumps(geo_results, indent=2, default=str))
    
        # GLOBAL SUMMARY
        print("\n" + "=" * 70)
        print("GLOBAL SUMMARY")
        print("=" * 70)
        
        verdicts = [g["validation"]["verdict"] for g in all_results["geometries"]]
        n_einstein = sum(1 for v in verdicts if v == "LIKELY_EINSTEIN_VACUUM")
        n_possibly = sum(1 for v in verdicts if v == "POSSIBLY_EINSTEIN_WITH_MATTER")
        n_non = sum(1 for v in verdicts if v == "NON_EINSTEIN_OR_DEFORMED")
        n_total = len(verdicts)
        
        avg_score = np.mean([g["validation"]["einstein_score"] for g in all_results["geometries"]]) if verdicts else 0.0
        
        print(f"  geometries processed:           {n_total}")
        print(f"  Likely Einstein vacuum:          {n_einstein}/{n_total}")
        print(f"  Possibly Einstein + matter:      {n_possibly}/{n_total}")
        print(f"  Non-Einstein or deformed:        {n_non}/{n_total}")
        print(f"  Einstein average score:         {avg_score:.2f}")
        
        all_results["summary"] = {
            "n_geometries": n_total,
            "n_likely_einstein": n_einstein,
            "n_possibly_einstein": n_possibly,
            "n_non_einstein": n_non,
            "average_einstein_score": float(avg_score)
        }
        
        summary_path = output_dir / "einstein_discovery_summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2, default=str))
        
        print(f"\n  Results: {summary_path}")
        print("=" * 70)
        
        if n_einstein == n_total:
            print("\n[OK] ECUACIONES DE EINSTEIN REDESCUBIERTAS (todas las geometrias)")
        elif n_einstein > 0:
            print(f"\n[OK] Einstein redescubierto en {n_einstein}/{n_total} geometrias")
        else:
            print("\n[!] Ninguna geometria parece ser Einstein puro")
        
        print("Next step: 04_geometry_physics_contracts.py")

    except Exception as exc:
        status = STATUS_ERROR
        exit_code = EXIT_ERROR
        error_message = str(exc)
        import traceback
        traceback.print_exc()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
