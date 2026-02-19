#!/usr/bin/env python3
# 07_emergent_lambda_sl_dictionary.py
# CUERDAS — Bloque C: Diccionario emergente λ_SL ↔ Δ (con contratos por régimen)
#
# OBJETIVO
#   Aprender una relación emergente entre el espectro escalar en el bulk (λ_SL)
#   y los exponentes UV/Δ, utilizando:
#     - Un modelo suave (p.ej. KAN) para aproximar la relación.
#     - PySR (u otro SR) para destilar una forma simbólica compacta.
#
# ENTRADAS
#   - runs/<experiment>/06_build_bulk_eigenmodes_dataset/bulk_modes_dataset.csv
#
# SALIDAS (V3)
#   runs/<experiment>/07_emergent_lambda_sl_dictionary/
#     lambda_sl_dictionary_report.json
#     lambda_sl_dictionary_pareto.csv
#     stage_summary.json
#
# RELACIÓN CON OTROS SCRIPTS
#   - Consume el dataset generado por: 06_build_bulk_eigenmodes_dataset.py
#   - Sus resultados se usan en: 09_real_data_and_dictionary_contracts.py
#
# HONESTIDAD
#   - No se fuerza la fórmula Δ(Δ-d) ni se inyectan diccionarios conocidos.
#   - Cualquier comparación con fórmulas teóricas se realiza posteriormente.
#   - La comparación con teoría solo se activa con --compare-theory.
#   - Evaluación por regímenes para detectar mezcla de escalas engañosa.
#
# MIGRADO A V3: 2024-12-23
# PATCH ROUTING_CONTRACT: 2024-12-27

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

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
    from cuerdas_io import update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False


# =============================================================================
# CONTRATOS POR RÉGIMEN: Definición de umbrales
# =============================================================================

@dataclass
class RegimeContractConfig:
    """Configuración de contratos por régimen de lambda_sl."""
    regime_lo_threshold: float = 1.0     # lambda_sl < 1
    regime_hi_threshold: float = 10.0    # lambda_sl > 10
    max_mre_for_pass: float = 0.5        # MRE < 50% para PASS
    mae_must_beat_baseline: bool = True  # MAE < MAE_baseline para PASS
    min_samples_for_regime: int = 5      # Mínimo de muestras para evaluar régimen


@dataclass
class DiscoveryConfig:
    """Configuración para el descubrimiento de relaciones emergentes."""
    niterations: int = 200
    populations: int = 30
    ncycles_per_iteration: int = 1000
    maxsize: int = 30
    features: Tuple[str, ...] = ("Delta", "d")
    target: str = "lambda_sl_emergent"
    binary_operators: Tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: Tuple[str, ...] = ("square", "sqrt", "exp", "log")
    binary_ops_minimal: Tuple[str, ...] = ("+", "-", "*", "/")
    unary_ops_minimal: Tuple[str, ...] = ("square", "sqrt")
    complexity_of_constants: float = 1.0
    complexity_of_variables: float = 1.0
    parsimony: float = 0.003
    annealing: bool = True
    test_split_ratio: float = 0.2
    random_state: int = 42


# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================

def load_emergent_data(input_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga datos de eigenmodos desde CSV o JSON.
    
    Returns:
        df: DataFrame con columnas Delta, d, lambda_sl_emergent
        metadata: Diccionario con metadatos del archivo
    """
    metadata = {
        "source_file": str(input_path),
        "load_timestamp": datetime.now().isoformat(),
        "suspicious_methods_found": []
    }
    
    suffix = input_path.suffix.lower()
    
    if suffix == ".csv":
        df = pd.read_csv(input_path)
        metadata["format"] = "csv"
    elif suffix == ".json":
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if "operators" in data:
            # Formato v2/v3
            df = pd.DataFrame(data["operators"])
            metadata["format"] = "json_v2"
            metadata["source_metadata"] = {k: v for k, v in data.items() if k != "operators"}
        else:
            df = pd.DataFrame(data)
            metadata["format"] = "json_legacy"
    else:
        raise ValueError(f"Formato no soportado: {suffix}")
    
    # Normalizar nombres de columnas
    column_mapping = {
        "lambda_sl": "lambda_sl_emergent",
        "eigenvalue": "lambda_sl_emergent",
        "m2L2": "lambda_sl_emergent",  # Legacy
        "delta": "Delta",
        "Delta_boundary": "Delta",
        "Delta_UV": "Delta",  # IO CONTRACT: 06 genera Delta_UV
        "dimension": "d",
        "spacetime_dim": "d",
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Verificar columnas requeridas
    required = ["Delta", "d", "lambda_sl_emergent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {list(df.columns)}")
    
    # Detectar métodos sospechosos
    if "method" in df.columns:
        suspicious = ["theoretical", "injected", "forced", "assumed"]
        mask = df["method"].str.lower().str.contains("|".join(suspicious), na=False)
        if mask.any():
            methods = df.loc[mask, "method"].unique().tolist()
            metadata["suspicious_methods_found"] = methods
            print(f"[WARN] Métodos sospechosos detectados: {methods}")
    
    metadata["total_operators"] = len(df)
    metadata["columns"] = list(df.columns)
    
    return df, metadata


def prepare_training_data(
    df: pd.DataFrame,
    config: DiscoveryConfig,
    test_split: float = 0.2,
    filter_suspicious: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepara datos para entrenamiento de PySR.
    
    Returns:
        X_train, y_train, X_test, y_test, test_df
    """
    df_clean = df.copy()
    
    # Filtrar métodos sospechosos si se solicita
    if filter_suspicious and "method_is_suspicious" in df_clean.columns:
        df_clean = df_clean[~df_clean["method_is_suspicious"]]
    
    # Eliminar NaN/Inf
    for col in ["Delta", "d", "lambda_sl_emergent"]:
        df_clean = df_clean[df_clean[col].notna()]
        df_clean = df_clean[np.isfinite(df_clean[col])]
    
    # Split train/test
    np.random.seed(config.random_state)
    n = len(df_clean)
    indices = np.random.permutation(n)
    n_test = int(n * test_split)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    train_df = df_clean.iloc[train_idx]
    test_df = df_clean.iloc[test_idx]
    
    X_train = train_df[list(config.features)].values
    y_train = train_df[config.target].values
    X_test = test_df[list(config.features)].values
    y_test = test_df[config.target].values
    
    print(f"   Datos de entrenamiento: {len(X_train)} muestras")
    print(f"   Datos de test: {len(X_test)} muestras")
    
    return X_train, y_train, X_test, y_test, test_df


# =============================================================================
# DESCUBRIMIENTO SIMBÓLICO
# =============================================================================

def discover_emergent_relation(
    X: np.ndarray,
    y: np.ndarray,
    config: DiscoveryConfig,
    use_minimal_ops: bool = False
) -> Optional[PySRRegressor]:
    """
    Ejecuta PySR para descubrir relación simbólica.
    """
    if not HAS_PYSR:
        print("[ERROR] PySR no disponible")
        return None
    
    binary_ops = list(config.binary_ops_minimal if use_minimal_ops else config.binary_operators)
    unary_ops = list(config.unary_ops_minimal if use_minimal_ops else config.unary_operators)
    
    print(f"\n   Configuración PySR:")
    print(f"   - Iteraciones: {config.niterations}")
    print(f"   - Operadores binarios: {binary_ops}")
    print(f"   - Operadores unarios: {unary_ops}")
    print(f"   - Tamaño máximo: {config.maxsize}")
    
    model = PySRRegressor(
        niterations=config.niterations,
        populations=config.populations,
        ncycles_per_iteration=config.ncycles_per_iteration,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        maxsize=config.maxsize,
        complexity_of_constants=config.complexity_of_constants,
        complexity_of_variables=config.complexity_of_variables,
        parsimony=config.parsimony,
        annealing=config.annealing,
        random_state=config.random_state,
        deterministic=True,
        parallelism='serial',
        verbosity=1,
    )
    
    print("\n   Ejecutando regresión simbólica...")
    try:
        model.fit(X, y, variable_names=list(config.features))
    except Exception as e:
        print(f"[ERROR] Fallo en PySR: {e}")
        return None
    
    return model


# =============================================================================
# EVALUACIÓN POR RÉGIMEN
# =============================================================================

def evaluate_by_regime(
    model: PySRRegressor,
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    config: RegimeContractConfig
) -> Dict[str, Any]:
    """
    Evalúa el modelo por régimen de lambda_sl.
    """
    y_pred = model.predict(X)
    
    results = {
        "global": compute_metrics(y, y_pred),
        "regimes": {},
        "contract_status": "UNKNOWN"
    }
    
    # Definir regímenes
    regimes = {
        "lo": y < config.regime_lo_threshold,
        "mid": (y >= config.regime_lo_threshold) & (y <= config.regime_hi_threshold),
        "hi": y > config.regime_hi_threshold
    }
    
    all_pass = True
    any_evaluated = False
    
    for regime_name, mask in regimes.items():
        n_samples = mask.sum()
        
        if n_samples < config.min_samples_for_regime:
            results["regimes"][regime_name] = {
                "n_samples": int(n_samples),
                "status": "SKIPPED",
                "reason": f"Menos de {config.min_samples_for_regime} muestras"
            }
            continue
        
        any_evaluated = True
        y_regime = y[mask]
        y_pred_regime = y_pred[mask]
        
        metrics = compute_metrics(y_regime, y_pred_regime)
        
        # Evaluar contrato
        mre = metrics.get("mre", float('inf'))
        baseline_mae = np.mean(np.abs(y_regime - np.mean(y_regime)))
        mae = metrics.get("mae", float('inf'))
        
        passes_mre = mre < config.max_mre_for_pass
        passes_baseline = mae < baseline_mae if config.mae_must_beat_baseline else True
        
        status = "PASS" if (passes_mre and passes_baseline) else "FAIL"
        if status == "FAIL":
            all_pass = False
        
        results["regimes"][regime_name] = {
            "n_samples": int(n_samples),
            "metrics": metrics,
            "baseline_mae": float(baseline_mae),
            "passes_mre": passes_mre,
            "passes_baseline": passes_baseline,
            "status": status
        }
    
    if not any_evaluated:
        results["contract_status"] = "INCONCLUSIVE"
    elif all_pass:
        results["contract_status"] = "PASS"
    else:
        results["contract_status"] = "FAIL"
    
    return results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de evaluación."""
    metrics = {}
    
    try:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    except:
        metrics["r2"] = float('nan')
    
    try:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    except:
        metrics["mae"] = float('nan')
    
    try:
        # MRE: Mean Relative Error
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs(y_true - y_pred) / np.abs(y_true)
            relative_errors = relative_errors[np.isfinite(relative_errors)]
            metrics["mre"] = float(np.mean(relative_errors)) if len(relative_errors) > 0 else float('nan')
    except:
        metrics["mre"] = float('nan')
    
    try:
        corr, pval = pearsonr(y_true, y_pred)
        metrics["pearson"] = float(corr)
        metrics["pearson_pval"] = float(pval)
    except:
        metrics["pearson"] = float('nan')
        metrics["pearson_pval"] = float('nan')
    
    return metrics


# =============================================================================
# COMPARACIÓN POST-HOC CON TEORÍA (solo si --compare-theory)
# =============================================================================

def compare_with_theory(
    df: pd.DataFrame,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Comparación POST-HOC con fórmula teórica Δ(Δ-d).
    IMPORTANTE: Esto es solo validación, nunca entra en entrenamiento.
    """
    Delta = df["Delta"].values
    d = df["d"].values
    y_true = df["lambda_sl_emergent"].values
    
    # Fórmula teórica: λ_SL = Δ(Δ - d)
    y_theory = Delta * (Delta - d)
    
    theory_metrics = compute_metrics(y_true, y_theory)
    pred_metrics = compute_metrics(y_true, y_pred)
    
    # Determinar compatibilidad
    compatible = theory_metrics.get("r2", 0) > 0.9 and pred_metrics.get("r2", 0) > 0.9
    
    return {
        "theory_formula": "Delta * (Delta - d)",
        "theory_r2": theory_metrics.get("r2"),
        "theory_mae": theory_metrics.get("mae"),
        "discovered_r2": pred_metrics.get("r2"),
        "discovered_mae": pred_metrics.get("mae"),
        "compatible_with_maldacena": compatible,
        "note": "POST-HOC comparison only, not used in training"
    }


# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================

def save_results(
    model: PySRRegressor,
    config: DiscoveryConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: Path,
    use_minimal_ops: bool,
    regime_config: RegimeContractConfig,
    compare_theory: bool
) -> Dict[str, Any]:
    """Guarda todos los resultados."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener mejor ecuación
    best_eq = str(model.sympy())
    best_complexity = int(model.equations_.iloc[-1]['complexity'])
    best_loss = float(model.equations_.iloc[-1]["loss"])
    
    # Métricas en test
    y_pred_test = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    # Evaluación por régimen
    regime_results = evaluate_by_regime(model, X_test, y_test, test_df, regime_config)
    
    # Comparación con teoría (solo si se solicita)
    theory_comparison = {}
    if compare_theory:
        theory_comparison = compare_with_theory(test_df, y_pred_test)
    else:
        theory_comparison = {"enabled": False, "note": "Use --compare-theory to enable"}
    
    # Construir summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "version": "v3_routing_contract",
        "source_metadata": metadata,
        "config": {
            "niterations": config.niterations,
            "maxsize": config.maxsize,
            "features": list(config.features),
            "target": config.target,
            "use_minimal_ops": use_minimal_ops,
            "random_state": config.random_state,
        },
        "regime_config": asdict(regime_config),
        "discovery_results": {
            "best_equation": best_eq,
            "best_complexity": best_complexity,
            "best_loss": best_loss,
            "test_metrics": test_metrics,
        },
        "feature_mapping": {
            "x_mapping": {f"x{i}": name for i, name in enumerate(config.features)},
        },
        "metrics_by_regime": regime_results["regimes"],
        "contract_status": regime_results["contract_status"],
        "theory_comparison": theory_comparison,
    }
    
    # Guardar JSON principal
    report_path = output_dir / "lambda_sl_dictionary_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n   Guardado: {report_path}")
    
    # Guardar Pareto front
    try:
        pareto_path = output_dir / "lambda_sl_dictionary_pareto.csv"
        model.equations_.to_csv(pareto_path, index=False)
        print(f"   Guardado: {pareto_path}")
    except Exception as e:
        print(f"   [WARN] No se pudo guardar Pareto: {e}")
    
    return summary


# =============================================================================
# ROUTING CONTRACT: Validación de conflictos
# =============================================================================

def validate_routing_args(args) -> Tuple[bool, str]:
    """
    Valida que no haya conflictos entre --experiment y --output-dir.
    Según ROUTING_CONTRACT: --experiment es la fuente de verdad.
    
    Returns:
        (is_valid, error_message)
    """
    has_experiment = getattr(args, 'experiment', None) is not None
    has_output_dir = getattr(args, 'output_dir', None) is not None
    
    if has_experiment and has_output_dir:
        return False, (
            "CONFLICTO: --experiment y --output-dir son mutuamente excluyentes.\n"
            "  --experiment es la fuente de verdad (ROUTING_CONTRACT).\n"
            "  --output-dir está DEPRECATED.\n"
            "  Usa solo --experiment."
        )
    
    if has_output_dir:
        warnings.warn(
            "--output-dir está DEPRECATED. Usa --experiment en su lugar.",
            DeprecationWarning,
            stacklevel=2
        )
    
    return True, ""


def resolve_input_file(args, ctx) -> Optional[Path]:
    """
    Resuelve el archivo de entrada según ROUTING_CONTRACT.
    
    Prioridad:
    1. --input-file explícito (DEPRECATED, con warning)
    2. --experiment → buscar en stage 06
    3. --run-dir legacy → buscar en bulk_eigenmodes/
    """
    # Prioridad 1: --input-file explícito
    if args.input_file:
        input_path = Path(args.input_file).resolve()
        if input_path.exists():
            warnings.warn(
                "--input-file está DEPRECATED. Usa --experiment en su lugar.",
                DeprecationWarning
            )
            return input_path
        else:
            print(f"[ERROR] Archivo especificado no existe: {input_path}")
            return None
    
    # Prioridad 2: V3 - buscar en stage 06
    if ctx:
        candidates = [
            ctx.run_root / "06_build_bulk_eigenmodes_dataset" / "bulk_modes_dataset.csv",
            ctx.run_root / "bulk_eigenmodes" / "bulk_modes_dataset.csv",  # alias legacy
        ]
        for candidate in candidates:
            if candidate.exists():
                print(f"[V3] Input desde stage 06: {candidate}")
                return candidate
        
        # No encontrado - dar error claro
        print(f"[ERROR] No se encontró bulk_modes_dataset.csv en:")
        for c in candidates:
            print(f"        - {c}")
        print(f"\n        Ejecuta primero: python 06_build_bulk_eigenmodes_dataset.py --experiment {ctx.experiment}")
        return None
    
    # Prioridad 3: --run-dir legacy
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        candidates = [
            run_dir / "bulk_eigenmodes" / "bulk_modes_dataset.csv",
            run_dir / "06_build_bulk_eigenmodes_dataset" / "bulk_modes_dataset.csv",
            run_dir / "bulk_eigenmodes" / "bulk_modes_dataset_v2.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        print(f"[ERROR] No se encontró dataset en {run_dir}")
        return None
    
    return None


def resolve_output_dir(args, ctx) -> Optional[Path]:
    """
    Resuelve el directorio de salida según ROUTING_CONTRACT.
    
    Prioridad:
    1. --experiment → ctx.stage_dir (V3)
    2. --output-dir explícito (DEPRECATED)
    3. --run-dir / emergent_dictionary (legacy)
    """
    if ctx:
        return ctx.stage_dir
    
    if args.output_dir:
        return Path(args.output_dir).resolve()
    
    if args.run_dir:
        return Path(args.run_dir).resolve() / "emergent_dictionary"
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FASE XII.c v3: Diccionario Emergente λ_SL ↔ Δ (con contratos por régimen)"
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
    parser.add_argument("--input-file", type=str, default=None,
                        help="[DEPRECATED] Archivo de entrada. Usar --experiment.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="[DEPRECATED] Directorio de salida. Usar --experiment.")
    
    # Argumentos específicos del script
    parser.add_argument("--ops-minimal", action="store_true",
                        help="Usar operadores mínimos (+,-,*,/,square,sqrt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Número de iteraciones de PySR")
    parser.add_argument("--no-filter-suspicious", action="store_true",
                        help="No filtrar operadores con métodos sospechosos")
    parser.add_argument("--drop-suspicious", action="store_true",
                        help="Descartar filas con method_is_suspicious==True antes de entrenar")
    parser.add_argument("--force-continue", action="store_true",
                        help="Continuar aunque se detecten métodos sospechosos")
    
    # Argumentos para contratos por régimen
    parser.add_argument("--compare-theory", action="store_true",
                        help="Activar comparación post-hoc con Δ(Δ-d). OFF por defecto.")
    parser.add_argument("--regime-lo", type=float, default=1.0,
                        help="Umbral inferior para régimen 'lo' (default: lambda_sl < 1.0)")
    parser.add_argument("--regime-hi", type=float, default=10.0,
                        help="Umbral superior para régimen 'hi' (default: lambda_sl > 10.0)")
    parser.add_argument("--max-mre", type=float, default=0.5,
                        help="MRE máximo para PASS en contratos (default: 0.5 = 50%%)")
    
    args = parser.parse_args()
    
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
                stage_number="07",
                stage_slug="emergent_lambda_sl_dictionary"
            )
            print(f"[V3] Experiment: {ctx.experiment}")
            print(f"[V3] Stage dir: {ctx.stage_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESOLVER INPUT
    # ═══════════════════════════════════════════════════════════════════════
    input_path = resolve_input_file(args, ctx)
    
    if input_path is None:
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "no_input_file"})
        return 2
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESOLVER OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    output_dir = resolve_output_dir(args, ctx)
    
    if output_dir is None:
        print("[ERROR] No se especificó directorio de salida.")
        print("        Usar --experiment o --output-dir")
        return 2
    
    config = DiscoveryConfig(random_state=args.seed, niterations=args.iterations)
    
    regime_config = RegimeContractConfig(
        regime_lo_threshold=args.regime_lo,
        regime_hi_threshold=args.regime_hi,
        max_mre_for_pass=args.max_mre,
        mae_must_beat_baseline=True
    )
    
    print("=" * 80)
    print("FASE XII.c v3 - DICCIONARIO HOLOGRÁFICO EMERGENTE")
    print("Nomenclatura honesta: λ_SL (autovalores Sturm–Liouville)")
    print("Con contratos por régimen y --compare-theory OFF por defecto")
    print("=" * 80)
    
    if not input_path.exists():
        print(f"[ERROR] Archivo no encontrado: {input_path}")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "input_not_found"})
        return 2
    
    print(f"\n   Archivo de entrada: {input_path}")
    
    try:
        df, metadata = load_emergent_data(input_path)
    except Exception as e:
        print(f"[ERROR] Fallo cargando datos: {e}")
        if ctx:
            ctx.write_summary(status="ERROR", counts={"error": str(e)})
        return 3
    
    # Filtrar métodos sospechosos si se solicitó
    if args.drop_suspicious and "method_is_suspicious" in df.columns:
        n_before = len(df)
        df = df[~df["method_is_suspicious"]].copy()
        n_after = len(df)
        if n_before > n_after:
            print(f"\n   >> Drop suspicious: {n_before - n_after} operadores eliminados, quedan {n_after}.")
            metadata["dropped_suspicious"] = n_before - n_after
    
    n_suspicious = len(metadata.get("suspicious_methods_found", []))
    if n_suspicious > 0 and not args.force_continue:
        print(f"\n   Se encontraron {n_suspicious} métodos sospechosos.")
        response = input("   ¿Continuar? (s/N): ")
        if response.lower() != 's':
            print("   Abortado por el usuario.")
            if ctx:
                ctx.write_summary(status="INCOMPLETE", counts={"error": "user_abort"})
            return 1
    
    try:
        X_train, y_train, X_test, y_test, test_df = prepare_training_data(
            df, config, test_split=config.test_split_ratio,
            filter_suspicious=not args.no_filter_suspicious
        )
    except Exception as e:
        print(f"[ERROR] Fallo preparando datos: {e}")
        if ctx:
            ctx.write_summary(status="ERROR", counts={"error": str(e)})
        return 3
    
    if not HAS_PYSR:
        print("   ERROR: PySR no disponible. Instalar con: pip install pysr")
        if ctx:
            ctx.write_summary(status="ERROR", counts={"error": "pysr_not_available"})
        return 1
    
    model = discover_emergent_relation(X_train, y_train, config, use_minimal_ops=args.ops_minimal)
    
    if model is None:
        if ctx:
            ctx.write_summary(status="ERROR", counts={"error": "model_training_failed"})
        return 1
    
    summary = save_results(
        model, config, X_train, y_train, X_test, y_test,
        test_df, metadata, output_dir, args.ops_minimal,
        regime_config, args.compare_theory
    )
    
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    best_eq = summary["discovery_results"]["best_equation"]
    test_metrics = summary["discovery_results"]["test_metrics"]
    theory_comp = summary["theory_comparison"]
    
    print(f"\n   ECUACIÓN DESCUBIERTA: {best_eq}")
    print(f"\n   x_mapping: {summary['feature_mapping']['x_mapping']}")
    print(f"\n   MÉTRICAS EN TEST (GLOBALES):")
    print(f"   - R²: {test_metrics.get('r2', 'N/A'):.4f}")
    print(f"   - MAE: {test_metrics.get('mae', 'N/A'):.4f}")
    print(f"   - Pearson: {test_metrics.get('pearson', 'N/A'):.4f}")
    
    contract_status = summary.get("contract_status", "UNKNOWN")
    print(f"\n   ESTADO DE CONTRATOS POR RÉGIMEN: {contract_status}")
    
    if contract_status == "FAIL":
        print(f"   ⚠ El diccionario NO generaliza bien en todos los regímenes de λ_SL.")
        print(f"   Revisar metrics_by_regime en el JSON para detalles.")
    elif contract_status == "PASS":
        print(f"   ✓ El diccionario pasa contratos en todos los regímenes evaluados.")
    else:
        print(f"   ? Estado inconcluso - revisar regímenes individuales.")
    
    if args.compare_theory:
        print(f"\n   COMPARACIÓN A POSTERIORI CON TEORÍA (--compare-theory activado):")
        print(f"   - Fórmula teórica: λ_SL = Δ(Δ - d)")
        print(f"   - R² teórico: {theory_comp.get('theory_r2', 'N/A')}")
        print(f"   - Compatible: {theory_comp.get('compatible_with_maldacena', 'N/A')}")
    else:
        print(f"\n   Comparación con teoría Δ(Δ-d): DESHABILITADA (usar --compare-theory)")
    
    print(f"\n   Resultados en: {output_dir.absolute()}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # V3: Registrar artefactos y escribir summary
    # ═══════════════════════════════════════════════════════════════════════
    if ctx:
        report_file = output_dir / "lambda_sl_dictionary_report.json"
        ctx.record_artifact("dictionary_report", report_file)
        ctx.record_artifact("input_file", input_path)
        
        ctx.write_summary(
            status="OK" if contract_status in ["PASS", "INCONCLUSIVE"] else "WARNING",
            counts={
                "total_operators": metadata.get("total_operators", 0),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "contract_status": contract_status,
                "test_r2": test_metrics.get("r2", 0.0),
            }
        )
        ctx.write_manifest()
        print(f"[V3] stage_summary.json escrito")
    
    # Legacy: actualizar run_manifest
    elif args.run_dir and HAS_CUERDAS_IO:
        try:
            run_dir = Path(args.run_dir).resolve()
            report_file = output_dir / "lambda_sl_dictionary_report.json"
            
            # Usar safe_relpath para evitar crash
            def safe_relpath(path: Path, base: Path) -> str:
                try:
                    return str(path.relative_to(base))
                except ValueError:
                    return str(path)
            
            update_run_manifest(
                run_dir,
                {
                    "emergent_dictionary_dir": safe_relpath(output_dir, run_dir),
                    "dictionary_report": safe_relpath(report_file, run_dir),
                }
            )
            print(f"   Manifest actualizado (legacy)")
        except Exception as e:
            print(f"   [WARN] No se pudo actualizar manifest: {e}")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
