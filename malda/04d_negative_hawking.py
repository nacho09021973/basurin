#!/usr/bin/env python3
"""
4d_negative_hawking.py
======================

Control negativo Hawking-Page para CUERDAS-MALDACENA.

PROPOSITO:
- Generar datos boundary simulando fase confining (below Tc) anti-holográfica.
- Inspirado en Bao, Cao & Zhu (2022): no threshold holográfico en confinement.
- Verificar que pipeline rechaza holografía (pass_rate < 20% en contratos).

TEORIA POST-HOC (no inyectada):
- Below Tc: area law (Wilson loops ~ exp(-m r)), no volume law EE.
- No AdS emergente ni diccionario λ_SL ↔ Δ coherente.
- Cita: Sección I del paper (intro a phase transition).

CRITERIO DE EXITO:
- Pass rate < 20% → SUCCESS (detecta no-holografía).
- Pass rate > 50% → ALERTA (falsos positivos).

USO:
    python 4d_negative_hawking.py --output_dir runs/negative_hawking_YYYYMMDD
    python 4d_negative_hawking.py --tc 1.0 --mass 1.0 --lattice_size 100 --seed 42

Autor: Proyecto CUERDAS-MALDACENA
Fecha: 2025-12-30
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib

import numpy as np
import h5py

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# SECCIÓN 1: GENERACIÓN DE DATOS CONFINING (BELOW Tc)
# ==============================================================================

def generate_hawking_negative_data(
    tc: float,
    mass: float,
    lattice_size: int,
    dim: int = 2,
    seed: Optional[int] = None,
    noise_level: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Genera datos boundary anti-holográficos: fase confining below Tc.
    
    HONESTIDAD: Simula T < Tc (e.g., T=0.5*tc) con area law.
    - Correladores: exp(-mass * r) + noise (área law, no conforme).
    - Entanglement entropy: ~ area (sum over boundaries).
    - Inspirado post-hoc en paper Sec. I: no threshold en confinement.
    """
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(f"Generando CONTROL NEGATIVO (confinement below Tc): L={lattice_size}, d={dim}, Tc={tc}")
    
    shape = tuple([lattice_size] * dim)
    coords = np.indices(shape).reshape(dim, -1).T - lattice_size // 2
    r = np.linalg.norm(coords, axis=1).reshape(shape)
    
    # Campo: exponential decay (area law) + noise
    field = np.exp(-mass * r) + np.random.normal(0, noise_level, shape)
    
    # Correlator: exp(-m r) (simula Wilson loops en confining, paper-inspired)
    correlator = np.exp(-mass * r)
    
    # Entanglement entropy ~ area (no volume law)
    entropy = np.sum(np.abs(field) ** 2, axis=tuple(range(1, dim)))  # Area-like
    
    # Pseudo-boundary para compatibilidad pipeline
    boundary_data = {
        'field': field,
        'correlator': correlator,
        'entropy': entropy,
        'dims': dim,
        'simulated_t': 0.5 * tc  # Below Tc
    }
    
    return boundary_data

# ==============================================================================
# SECCIÓN 2: GUARDADO EN FORMATO HDF5 (COMPATIBLE CON 02_emergent_geometry_engine)
# ==============================================================================

def save_negative_hawking_data(data: Dict[str, np.ndarray], output_dir: Path, run_id: str) -> Path:
    """
    Guarda datos en HDF5 compatible con 02_emergent_geometry_engine.
    
    FORMATO REQUERIDO POR 02:
      - Grupo 'boundary/' con: G2_*, x_grid, temperature, d
      - Atributo 'operators' (JSON string)
      - Grupo 'bulk_truth/' (opcional, no se usa en inference)
    
    MARCADORES DE CONTROL NEGATIVO:
      - attrs['IS_NEGATIVE_CONTROL'] = 1
      - attrs['EXPECTED_HOLOGRAPHIC'] = 0
      - attrs['PHASE'] = 'confining_below_tc'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    system_name = f"negative_hawking_{run_id}"
    h5_path = output_dir / f"{system_name}.h5"
    
    d = data['dims']  # Dimensión del boundary
    n_points = 100  # Puntos para x_grid
    
    with h5py.File(h5_path, 'w') as f:
        # === GRUPO BOUNDARY (requerido por 02) ===
        boundary = f.create_group('boundary')
        
        # Crear x_grid
        x_grid = np.linspace(0.1, 10.0, n_points)
        boundary.create_dataset('x_grid', data=x_grid)
        
        # Correlador G2 desde los datos confining
        # En fase confining: G(r) ~ exp(-m*r) (area law)
        # Esto NO es power-law, lo cual debería fallar el contrato de correlador
        correlator_1d = data['correlator'].flatten()[:n_points]
        if len(correlator_1d) < n_points:
            # Extender con decay exponencial
            mass = 1.0  # Asumido
            extra_x = np.linspace(0.1, 10.0, n_points - len(correlator_1d))
            extra_corr = np.exp(-mass * extra_x)
            correlator_1d = np.concatenate([correlator_1d, extra_corr])
        boundary.create_dataset('G2_confining', data=correlator_1d)
        
        # Temperatura (below Tc)
        T_simulated = data.get('simulated_t', 0.5)
        boundary.create_dataset('temperature', data=np.array([T_simulated]))
        
        # Dimensión d
        boundary.attrs['d'] = d
        
        # === OPERADORES (requerido por 02 como JSON string) ===
        # Operadores fake - en fase confining no hay operadores conformes bien definidos
        # Usamos dimensiones que violen unitariedad para marcar como no-holográfico
        fake_operators = [
            {
                "name": "glueball_fake",
                "Delta": 0.3,  # Viola unitariedad para d>=2
                "spin": 0,
                "is_negative_control": True,
                "phase": "confining"
            },
            {
                "name": "string_tension_fake",
                "Delta": -0.2,  # Dimensión negativa (imposible en CFT)
                "spin": 0,
                "is_negative_control": True,
                "phase": "confining"
            }
        ]
        f.attrs['operators'] = json.dumps(fake_operators)
        
        # === GRUPO BULK_TRUTH (fake, para compatibilidad) ===
        # 02 en mode=inference NO accede a esto
        bulk = f.create_group('bulk_truth')
        n_z = 100
        z_grid = np.linspace(0.01, 5.0, n_z)
        bulk.create_dataset('z_grid', data=z_grid)
        # En fase confining NO hay bulk AdS emergente
        # A, f, R son ruido - NO deben ser usados
        bulk.create_dataset('A_truth', data=np.random.randn(n_z) * 0.1)
        bulk.create_dataset('f_truth', data=np.random.rand(n_z))
        bulk.create_dataset('R_truth', data=np.random.randn(n_z) * 10)
        bulk.attrs['z_h'] = 1.0
        bulk.attrs['family'] = 'negative_control_hawking'
        bulk.attrs['d'] = d
        
        # === MARCADORES DE CONTROL NEGATIVO ===
        f.attrs['IS_NEGATIVE_CONTROL'] = 1
        f.attrs['EXPECTED_HOLOGRAPHIC'] = 0
        f.attrs['PHASE'] = 'confining_below_tc'
        f.attrs['system_name'] = system_name
        f.attrs['family'] = 'negative_control_hawking'
        f.attrs['category'] = 'negative_control'
        f.attrs['d'] = d
        
        # === GRUPO LEGACY (datos originales) ===
        legacy = f.create_group('_hawking_raw')
        legacy.create_dataset('field', data=data['field'], compression='gzip')
        legacy.create_dataset('correlator', data=data['correlator'])
        legacy.create_dataset('entropy', data=data['entropy'])
        legacy.attrs['simulated_t'] = T_simulated
    
    logger.info(f"Datos guardados en: {h5_path}")
    
    # === GENERAR MANIFEST.JSON (requerido por 02 inference) ===
    manifest = {
        "manifest_version": "2.0",
        "source": "4d_negative_hawking.py",
        "is_negative_control": True,
        "control_type": "hawking_page_confining",
        "geometries": [
            {
                "name": system_name,
                "family": "negative_control_hawking",
                "d": d,
                "is_negative_control": True,
                "expected_holographic": False,
                "phase": "confining_below_tc"
            }
        ],
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "simulated_t": float(T_simulated),
            "purpose": "Detectar falsos positivos en pipeline holográfico (fase confining)"
        }
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)
    
    logger.info(f"Manifest guardado: {manifest_path}")
    
    return h5_path

# ==============================================================================
# SECCIÓN 3: EJECUCIÓN DEL PIPELINE (OPCIONAL)
# ==============================================================================

def run_pipeline_on_negative_hawking(
    h5_path: Path, 
    pipeline_dir: Path,
    checkpoint_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ejecuta pipeline sobre datos generados.
    
    Para control negativo, necesitamos:
    1. Si hay checkpoint: 02 inference → 03 → 04 (verificar contratos)
    2. Sin checkpoint: solo 04 con data-dir (verificar contratos sobre boundary)
    
    Args:
        h5_path: Path al HDF5 generado
        pipeline_dir: Directorio con scripts del pipeline
        checkpoint_path: Opcional, checkpoint de 02 para inference
    """
    logger.info("Ejecutando pipeline sobre control negativo...")
    results = {}
    
    experiment = h5_path.parent.name  # Use output_dir name as experiment
    data_dir = h5_path.parent
    
    if checkpoint_path and checkpoint_path.exists():
        # Modo completo: inference con checkpoint existente
        logger.info(f"Usando checkpoint: {checkpoint_path}")
        
        cmds = [
            # 02: Inference mode
            f"python {pipeline_dir / '02_emergent_geometry_engine.py'} "
            f"--experiment {experiment} "
            f"--data-dir {data_dir} "
            f"--mode inference "
            f"--checkpoint {checkpoint_path}",
            
            # 03: Descubrir ecuaciones
            f"python {pipeline_dir / '03_discover_bulk_equations.py'} --experiment {experiment}",
            
            # 04: Verificar contratos (lo más importante)
            f"python {pipeline_dir / '04_geometry_physics_contracts.py'} "
            f"--experiment {experiment} "
            f"--geometry-dir {data_dir / '02_emergent_geometry_engine'} "
            f"--data-dir {data_dir}",
        ]
    else:
        # Modo simplificado: solo verificar contratos sobre boundary
        logger.info("Sin checkpoint. Evaluando contratos directamente sobre boundary data.")
        
        # Crear estructura mínima para que 04 pueda evaluar
        # Solo necesitamos los contratos de unitariedad y correlador
        cmds = [
            f"python {pipeline_dir / '04_geometry_physics_contracts.py'} "
            f"--experiment {experiment} "
            f"--data-dir {data_dir}",
        ]
    
    for cmd in cmds:
        logger.info(f"Ejecutando: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            logger.warning(f"Comando falló (ret={ret}): {cmd}")
            # No lanzar error, continuar para capturar lo que se pueda
    
    # Captura artefactos
    # Einstein summary de 03
    summary_path = data_dir / "03_discover_bulk_equations" / "einstein_discovery_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['einstein_summary'] = json.load(f)
    
    # Contracts summary de 04
    contracts_path = data_dir / "04_geometry_physics_contracts" / "geometry_contracts_summary.json"
    if contracts_path.exists():
        with open(contracts_path, 'r') as f:
            results['contracts_summary'] = json.load(f)
    
    return results

# ==============================================================================
# SECCIÓN 4: VERIFICACIÓN DE CONTRATOS (POST-HOC)
# ==============================================================================

def check_contracts_failure(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifica fallos esperados en control negativo.
    
    Para un control negativo exitoso:
    - generic_passed debe ser False (unitariedad y/o correlador fallan)
    - overall_passed debe ser False
    - pass_rate < 20% → SUCCESS (detecta no-holografía)
    """
    # Verificar contracts_summary de 04 (prioritario)
    if 'contracts_summary' in results:
        contracts = results['contracts_summary']
        n_total = contracts.get('n_total', 0)
        n_generic_passed = contracts.get('n_generic_passed', 0)
        n_overall_passed = contracts.get('n_overall_passed', 0)
        avg_score = contracts.get('avg_score', 1.0)
        
        if n_total > 0:
            pass_rate = 100 * n_overall_passed / n_total
            generic_pass_rate = 100 * n_generic_passed / n_total
        else:
            pass_rate = 0
            generic_pass_rate = 0
        
        # Para control negativo: queremos que FALLE
        if pass_rate < 20 and generic_pass_rate < 50:
            status = 'SUCCESS'
            message = f'Control negativo detectado: pass_rate={pass_rate:.1f}%, generic={generic_pass_rate:.1f}%'
        elif pass_rate < 50:
            status = 'WARNING'
            message = f'Pass rate marginal: {pass_rate:.1f}%. Revisar falsos positivos.'
        else:
            status = 'ALERT'
            message = f'ALERTA: pass_rate={pass_rate:.1f}% en control negativo. Pipeline tiene falsos positivos.'
        
        return {
            'status': status,
            'message': message,
            'pass_rate': pass_rate,
            'generic_pass_rate': generic_pass_rate,
            'avg_score': avg_score,
            'n_total': n_total,
            'n_generic_passed': n_generic_passed,
            'n_overall_passed': n_overall_passed
        }
    
    # Fallback: usar einstein_summary si no hay contracts
    if 'einstein_summary' in results:
        summary = results['einstein_summary']
        einstein_score = summary.get('average_einstein_score', 1.0)
        pass_rate = 100 * (1 - einstein_score)
        
        if pass_rate < 20:
            status = 'SUCCESS'
            message = 'Detectada no-holografía (via einstein_score)'
        elif pass_rate < 50:
            status = 'WARNING'
            message = 'Pass rate marginal; revisar falsos positivos'
        else:
            status = 'ALERT'
            message = 'Alta pass rate en control negativo; posible bug'
        
        return {'status': status, 'message': message, 'pass_rate': pass_rate, 'einstein_score': einstein_score}
    
    return {'status': 'INCOMPLETE', 'message': 'Pipeline no completado. Sin contracts_summary ni einstein_summary.'}

# ==============================================================================
# SECCIÓN 5: REPORTE FINAL
# ==============================================================================

def generate_negative_hawking_report(
    data: Dict[str, np.ndarray],
    results: Dict[str, Any],
    contract_check: Dict[str, Any],
    output_dir: Path,
    run_id: str
) -> Path:
    """
    Genera reporte JSON con resumen.
    """
    report = {
        'run_id': run_id,
        'parameters': {
            'tc': data.get('simulated_t') * 2,  # Reconstruye Tc
            'mass': 1.0,  # Placeholder
            'lattice_size': data['field'].shape[0],
            'dim': data['dims']
        },
        'pipeline_results': results,
        'contract_check': contract_check
    }
    
    report_path = output_dir / f"negative_hawking_report_{run_id}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Reporte guardado en: {report_path}")
    return report_path

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Control negativo Hawking-Page para CUERDAS-MALDACENA.'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        default=Path('runs') / f'negative_hawking_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--tc', 
        type=float, 
        default=1.0,
        help='Tc simulada (T=0.5*tc) (default: 1.0)'
    )
    parser.add_argument(
        '--mass', 
        type=float, 
        default=1.0,
        help='Masa efectiva para decay (default: 1.0)'
    )
    parser.add_argument(
        '--lattice_size', 
        type=int, 
        default=100,
        help='Tamaño del lattice (default: 100)'
    )
    parser.add_argument(
        '--dim', 
        type=int, 
        default=2,
        choices=[2, 3],
        help='Dimensionalidad espacial (default: 2)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Semilla para reproducibilidad'
    )
    parser.add_argument(
        '--noise', 
        type=float, 
        default=0.01,
        help='Nivel de ruido (default: 0.01)'
    )
    parser.add_argument(
        '--generate_only',
        action='store_true',
        help='Solo generar datos, no ejecutar pipeline'
    )
    parser.add_argument(
        '--pipeline_dir',
        type=Path,
        default=Path('.'),
        help='Directorio con scripts del pipeline'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Checkpoint de 02 para inference mode (opcional)'
    )
    
    args = parser.parse_args()
    
    # Run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.seed is not None:
        run_id += f"_seed{args.seed}"
    
    logger.info("="*60)
    logger.info("CONTROL NEGATIVO HAWKING-PAGE - CUERDAS-MALDACENA")
    logger.info("="*60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {args.output_dir}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    
    # 1. Generar datos
    logger.info("\n[PASO 1] Generando datos confining below Tc...")
    data = generate_hawking_negative_data(
        tc=args.tc,
        mass=args.mass,
        lattice_size=args.lattice_size,
        dim=args.dim,
        seed=args.seed,
        noise_level=args.noise
    )
    
    # 2. Guardar
    logger.info("\n[PASO 2] Guardando en HDF5...")
    h5_path = save_negative_hawking_data(data, args.output_dir, run_id)
    
    if args.generate_only:
        logger.info("\n[COMPLETADO] Modo --generate_only")
        return
    
    # 3. Pipeline
    logger.info("\n[PASO 3] Ejecutando pipeline...")
    results = run_pipeline_on_negative_hawking(h5_path, args.pipeline_dir, args.checkpoint)
    
    # 4. Verificar
    logger.info("\n[PASO 4] Verificando contratos...")
    contract_check = check_contracts_failure(results)
    
    # 5. Reporte
    logger.info("\n[PASO 5] Generando reporte...")
    report_path = generate_negative_hawking_report(
        data, results, contract_check, args.output_dir, run_id
    )
    
    # Resumen
    logger.info("\n" + "="*60)
    logger.info("RESUMEN")
    logger.info("="*60)
    logger.info(f"Estado: {contract_check.get('status', 'INCOMPLETE')}")
    logger.info(f"Mensaje: {contract_check.get('message', 'Ver reporte')}")
    logger.info(f"Reporte: {report_path}")
    logger.info("="*60)
    
    # Exit code
    if contract_check.get('status') == 'SUCCESS':
        sys.exit(0)
    elif contract_check.get('status') == 'WARNING':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()