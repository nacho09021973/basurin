#!/usr/bin/env python3
"""
04c_negative_controls.py
========================

Control negativo para validacion del pipeline CUERDAS-Maldacena.

PROPOSITO:
- Generar datos sinteticos que NO deberian producir holografia valida
- Verificar que el pipeline detecta la ausencia de estructura holografica
- Documentar el "fallo esperado" como evidencia de honestidad cientifica

IMPLEMENTACION (actualizada 2024-12-29):
Este script genera RUIDO BLANCO GAUSSIANO, no un campo escalar masivo
real que satisfaga Klein-Gordon. Esto es DELIBERADO:

- El ruido blanco tiene correlaciones delta (sin estructura)
- Es un control negativo MAS FUERTE que un campo masivo
- Si el pipeline "encuentra" holografia en ruido blanco, hay falsos positivos

TEORIA:
- Ruido blanco: G(r) ~ delta(r), no tiene simetria conforme
- No existe bulk AdS dual que emerja de datos sin correlaciones
- El diccionario holografico lambda_SL -> Delta no deberia converger

CRITERIO DE EXITO:
- Pass rate en contratos < 20% -> Sistema detecta ausencia de holografia
- Pass rate > 50% -> ALERTA: posible falso positivo, investigar

USO:
    python 04c_negative_controls.py --output_dir runs/negative_control_YYYYMMDD
    python 04c_negative_controls.py --mass 1.0 --lattice_size 100 --seed 42

Autor: Proyecto CUERDAS-Maldacena
Fecha: 2025-12-21
Actualizado: 2024-12-29 (honestidad: ruido blanco, no campo masivo)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
# SECCIÃƒâ€œN 1: GENERACIÃƒâ€œN DE DATOS ANTI-HOLOGRÃƒÂFICOS
# ==============================================================================

def generate_massive_scalar_flat_space(
    mass: float,
    lattice_size: int,
    dim: int = 2,
    seed: Optional[int] = None,
    noise_level: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Genera datos de un campo ANTI-HOLOGRAFICO para control negativo.
    
    NOTA DE HONESTIDAD (actualizada 2024-12-29):
    Este script genera RUIDO BLANCO GAUSSIANO, no un campo escalar masivo
    que satisfaga la ecuacion de Klein-Gordon. Esto es DELIBERADO:
    
    - El ruido blanco es MENOS correlacionado que un campo masivo real
    - Por tanto es un control negativo MAS FUERTE
    - Cualquier pipeline que "encuentre" holografia en ruido blanco tiene
      un problema serio de falsos positivos
    
    Propiedades del sistema generado:
    - Campo: ruido blanco gaussiano (correlaciones ~ delta)
    - Correlador: G(r) ~ delta(r) (no decaimiento exponencial)
    - NO conforme: no hay simetria de escala
    - NO holografico: no existe bulk AdS dual
    
    El parametro 'mass' se usa para pseudo-boundary pero NO afecta el campo.
    """
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(f"Generando CONTROL NEGATIVO (ruido blanco): L={lattice_size}, d={dim}")
    
    shape = tuple([lattice_size] * dim)
    
    # RUIDO BLANCO GAUSSIANO - deliberadamente NO es un campo fisico
    field = np.random.normal(0, 1, shape)
    field += np.random.normal(0, noise_level, shape)
    
    # Correlador REAL del campo (sera ~ delta para ruido blanco)
    correlator_2pt = _compute_correlator_from_field(field, dim)
    
    # Pseudo-boundary consistentes con correlador real
    pseudo_boundary_data = _create_pseudo_boundary_data(field, correlator_2pt, mass, lattice_size)
    
    metadata = {
        'type': 'white_noise_negative_control',
        'mass_parameter': mass,
        'lattice_size': lattice_size,
        'dimension': dim,
        'seed': seed,
        'noise_level': noise_level,
        'conformal': False,
        'expected_holographic': False,
        'is_physical_field': False,
        'purpose': 'negative_control_for_false_positive_detection',
        'generated_at': datetime.now().isoformat()
    }
    
    logger.info(f"  Campo: shape={field.shape} (ruido blanco)")
    logger.info(f"  Correlador: {len(correlator_2pt)} puntos")
    
    return {
        'field': field,
        'correlator_2pt': correlator_2pt,
        'pseudo_boundary_data': pseudo_boundary_data,
        'metadata': metadata
    }


def _compute_correlator_from_field(field: np.ndarray, dim: int) -> np.ndarray:
    """Calcula correlador REAL del campo usando FFT."""
    lattice_size = field.shape[0]
    max_r = lattice_size // 2
    
    correlator = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    field_fft = np.fft.fftn(field)
    power_spectrum = np.abs(field_fft) ** 2
    correlation_full = np.fft.ifftn(power_spectrum).real
    
    center = tuple([lattice_size // 2] * dim)
    
    for idx in np.ndindex(field.shape):
        r_squared = sum((i - c) ** 2 for i, c in zip(idx, center))
        r = int(np.sqrt(r_squared))
        if r < max_r:
            correlator[r] += correlation_full[idx]
            counts[r] += 1
    
    counts[counts == 0] = 1
    correlator /= counts
    
    if correlator[0] > 0:
        correlator /= correlator[0]
    
    return correlator


def _create_pseudo_boundary_data(
    field: np.ndarray,
    correlator: np.ndarray,
    mass: float,
    lattice_size: int
) -> Dict[str, np.ndarray]:
    """
    Crea pseudo-boundary CONSISTENTES con correlador real.
    
    El correlador G2['phi'] es el correlador REAL calculado del campo,
    no una formula analitica inventada. Esto asegura consistencia interna.
    """
    n_points = len(correlator)
    distances = np.arange(1, n_points + 1).astype(float)
    
    # Correlador REAL del campo (para ruido blanco: ~ delta)
    G2_real = correlator[1:n_points] if len(correlator) > n_points else correlator[1:]
    
    G2 = {
        'phi': G2_real,
        # Tambien incluir version fake exponencial para comparacion
        'phi_fake_exponential': np.exp(-mass * distances[:len(G2_real)]) / (distances[:len(G2_real)] ** 0.5 + 1e-10),
    }
    
    # Dimensiones que VIOLAN unitaridad (deliberado para control negativo)
    fake_dimensions = {
        'phi': 0.1,       # Violacion unitaridad (Delta < (d-2)/2)
        'phi_squared': -0.5,  # Dimension negativa (imposible)
    }
    
    return {
        'G2': G2,
        'fake_dimensions': fake_dimensions,
        'distances': distances,
        'correlator_type': 'white_noise_real',
        'warning': 'ANTI-HOLOGRAPHIC DATA - Expected to fail contracts'
    }



# ==============================================================================
# SECCIÓN 2: GUARDAR EN FORMATO COMPATIBLE CON PIPELINE (02_emergent_geometry_engine)
# ==============================================================================

def save_negative_control_data(
    data: Dict,
    output_dir: Path,
    run_id: str
) -> Path:
    """
    Guarda los datos del control negativo en formato HDF5 compatible
    con 02_emergent_geometry_engine.py.
    
    FORMATO REQUERIDO POR 02:
      - Grupo 'boundary/' con: G2_*, x_grid, temperature, d
      - Atributo 'operators' (JSON string)
      - Grupo 'bulk_truth/' (opcional, no se usa en inference)
    
    MARCADORES DE CONTROL NEGATIVO:
      - attrs['IS_NEGATIVE_CONTROL'] = 1
      - attrs['EXPECTED_HOLOGRAPHIC'] = 0
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre del sistema (sin extensión)
    system_name = f"negative_control_{run_id}"
    h5_path = output_dir / f"{system_name}.h5"
    
    logger.info(f"Guardando datos en: {h5_path}")
    
    meta = data['metadata']
    pseudo_boundary = data['pseudo_boundary_data']
    
    with h5py.File(h5_path, 'w') as f:
        # === GRUPO BOUNDARY (requerido por 02) ===
        boundary = f.create_group('boundary')
        
        # Correlador G2_phi (02 busca cualquier G2_*)
        G2_phi = pseudo_boundary['G2']['phi']
        boundary.create_dataset('G2_phi', data=G2_phi)
        
        # x_grid (02 usa 'x_grid', no 'distances')
        x_grid = pseudo_boundary['distances'][:len(G2_phi)]
        boundary.create_dataset('x_grid', data=x_grid)
        
        # Temperatura (02 busca 'temperature' o 'T')
        boundary.create_dataset('temperature', data=np.array([0.0]))
        
        # Dimensión d
        d = meta.get('dimension', 3)
        boundary.attrs['d'] = d
        
        # === OPERADORES (requerido por 02 como JSON string) ===
        # Operadores fake con dimensiones que violan unitariedad
        fake_operators = [
            {
                "name": "phi_fake",
                "Delta": pseudo_boundary['fake_dimensions']['phi'],
                "spin": 0,
                "is_negative_control": True
            },
            {
                "name": "phi2_fake", 
                "Delta": pseudo_boundary['fake_dimensions']['phi_squared'],
                "spin": 0,
                "is_negative_control": True
            }
        ]
        f.attrs['operators'] = json.dumps(fake_operators)
        
        # === GRUPO BULK_TRUTH (fake, para compatibilidad) ===
        # 02 en mode=inference NO accede a esto, pero lo creamos por si acaso
        bulk = f.create_group('bulk_truth')
        n_z = 100
        z_grid = np.linspace(0.01, 5.0, n_z)
        bulk.create_dataset('z_grid', data=z_grid)
        # A, f, R fake (ruido - NO deben ser usados)
        bulk.create_dataset('A_truth', data=np.random.randn(n_z) * 0.1)
        bulk.create_dataset('f_truth', data=np.random.rand(n_z))
        bulk.create_dataset('R_truth', data=np.random.randn(n_z) * 10)
        bulk.attrs['z_h'] = 1.0
        bulk.attrs['family'] = 'negative_control'
        bulk.attrs['d'] = d
        
        # === MARCADORES DE CONTROL NEGATIVO ===
        f.attrs['IS_NEGATIVE_CONTROL'] = 1
        f.attrs['EXPECTED_HOLOGRAPHIC'] = 0
        f.attrs['system_name'] = system_name
        
        # Metadata adicional
        for key, val in meta.items():
            if val is not None and key not in ['dimension']:
                if isinstance(val, bool):
                    f.attrs[key] = int(val)
                elif isinstance(val, (int, float, str)):
                    f.attrs[key] = val
        
        # === GRUPO LEGACY (para 04c interno) ===
        legacy = f.create_group('_negative_control_raw')
        legacy.create_dataset('field', data=data['field'], compression='gzip')
        legacy.create_dataset('correlator_2pt', data=data['correlator_2pt'])
    
    logger.info(f"  Datos guardados: {h5_path.stat().st_size / 1024:.1f} KB")
    
    # === GENERAR MANIFEST.JSON (requerido por 02 inference) ===
    manifest = {
        "manifest_version": "2.0",
        "source": "04c_negative_controls.py",
        "is_negative_control": True,
        "geometries": [
            {
                "name": system_name,
                "family": "negative_control",
                "d": d,
                "is_negative_control": True,
                "expected_holographic": False
            }
        ],
        "metadata": {
            "generated_at": meta.get('generated_at', datetime.now().isoformat()),
            "type": meta.get('type', 'white_noise_negative_control'),
            "purpose": "Detectar falsos positivos en pipeline holográfico"
        }
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)
    
    logger.info(f"  Manifest guardado: {manifest_path}")
    
    return h5_path


# ==============================================================================
# SECCIÃƒâ€œN 3: EJECUTAR PIPELINE Y VERIFICAR FALLOS
# ==============================================================================

def run_pipeline_on_negative_control(
    h5_path: Path,
    pipeline_scripts_dir: Path
) -> Dict[str, any]:
    """
    Ejecuta el pipeline sobre los datos del control negativo.
    
    NOTA: Esta funciÃƒÂ³n es un placeholder - la integraciÃƒÂ³n real
    depende de la estructura exacta del pipeline.
    
    Retorna dict con resultados de cada etapa.
    """
    logger.info("="*60)
    logger.info("EJECUTANDO PIPELINE SOBRE CONTROL NEGATIVO")
    logger.info("="*60)
    
    results = {
        'stages_run': [],
        'stages_failed': [],
        'contracts_checked': [],
        'contracts_passed': [],
        'contracts_failed': [],
    }
    
    # TODO: Integrar con pipeline real
    # Por ahora, documentamos la estructura esperada
    
    expected_stages = [
        ('02_emergent_geometry_engine.py', 'GeometrÃƒÂ­a emergente'),
        ('04_geometry_physics_contracts.py', 'Contratos fÃƒÂ­sicos'),
        ('05_scalar_field_solver.py', 'Solver escalar'),
        ('06_discover_symbolic_equations.py', 'Ecuaciones simbÃƒÂ³licas'),
    ]
    
    logger.warning("Pipeline no ejecutado - implementar integraciÃƒÂ³n")
    
    return results


def check_contracts_failure(
    results: Dict,
    expected_pass_rate: float = 0.2
) -> Dict[str, any]:
    """
    Verifica que los contratos fallen como se espera para datos no-hologrÃƒÂ¡ficos.
    
    Criterios:
    - pass_rate < 0.2: Ãƒâ€°XITO (sistema detecta no-holografÃƒÂ­a)
    - pass_rate 0.2-0.5: ADVERTENCIA (investigar)
    - pass_rate > 0.5: FALLO (posible falso positivo)
    """
    n_passed = len(results.get('contracts_passed', []))
    n_failed = len(results.get('contracts_failed', []))
    n_total = n_passed + n_failed
    
    if n_total == 0:
        return {
            'status': 'INCOMPLETE',
            'message': 'No se ejecutaron contratos',
            'pass_rate': None
        }
    
    pass_rate = n_passed / n_total
    
    if pass_rate < expected_pass_rate:
        status = 'SUCCESS'
        message = f'Sistema detectÃƒÂ³ correctamente ausencia de holografÃƒÂ­a (pass_rate={pass_rate:.2%})'
    elif pass_rate < 0.5:
        status = 'WARNING'
        message = f'Pass rate moderado ({pass_rate:.2%}) - investigar contratos especÃƒÂ­ficos'
    else:
        status = 'ALERT'
        message = f'POSIBLE FALSO POSITIVO: pass_rate={pass_rate:.2%} > 50%'
    
    return {
        'status': status,
        'message': message,
        'pass_rate': pass_rate,
        'n_passed': n_passed,
        'n_failed': n_failed,
        'n_total': n_total
    }


# ==============================================================================
# SECCIÃƒâ€œN 4: GENERAR REPORTE
# ==============================================================================

def generate_negative_control_report(
    data: Dict,
    results: Dict,
    contract_check: Dict,
    output_dir: Path,
    run_id: str
) -> Path:
    """
    Genera reporte markdown documentando el control negativo.
    """
    report_path = output_dir / f"negative_control_report_{run_id}.md"
    
    report = f"""# REPORTE DE CONTROL NEGATIVO

**Run ID:** {run_id}
**Fecha:** {datetime.now().isoformat()}
**Estado:** {contract_check.get('status', 'INCOMPLETE')}

---

## 1. DescripciÃƒÂ³n del Input

**Tipo:** Campo escalar masivo en espacio plano (flat space)

| ParÃƒÂ¡metro | Valor |
|-----------|-------|
| Masa (m) | {data['metadata'].get('mass', 'N/A')} |
| TamaÃƒÂ±o lattice | {data['metadata'].get('lattice_size', 'N/A')} |
| DimensiÃƒÂ³n | {data['metadata'].get('dimension', 'N/A')} |
| Seed | {data['metadata'].get('seed', 'N/A')} |
| Conforme | **NO** |
| HologrÃƒÂ¡fico esperado | **NO** |

### Por quÃƒÂ© este sistema NO es hologrÃƒÂ¡fico

1. **Sin simetrÃƒÂ­a conforme**: El tÃƒÂ©rmino de masa mÃ‚Â²Ãâ€ Ã‚Â² rompe la invariancia de escala
2. **Espacio plano**: No hay curvatura AdS que emerja naturalmente
3. **Correladores exponenciales**: G(r) ~ exp(-mr), no potencias como en CFT
4. **Dimensiones invÃƒÂ¡lidas**: Los "operadores" tienen ÃŽâ€ que violan unitaridad

---

## 2. Resultados del Pipeline

### Etapas ejecutadas
{_format_list(results.get('stages_run', ['(ninguna)']))}

### Etapas fallidas
{_format_list(results.get('stages_failed', ['(ninguna)']))}

---

## 3. VerificaciÃƒÂ³n de Contratos

| MÃƒÂ©trica | Valor |
|---------|-------|
| Contratos evaluados | {contract_check.get('n_total', 'N/A')} |
| Contratos pasados | {contract_check.get('n_passed', 'N/A')} |
| Contratos fallidos | {contract_check.get('n_failed', 'N/A')} |
| **Pass rate** | **{contract_check.get('pass_rate', 'N/A'):.2%}** |

### Contratos pasados (deberÃƒÂ­an ser pocos)
{_format_list(results.get('contracts_passed', ['(ninguno)']))}

### Contratos fallidos (esperados)
{_format_list(results.get('contracts_failed', ['(ninguno)']))}

---

## 4. ConclusiÃƒÂ³n

**{contract_check.get('message', 'AnÃƒÂ¡lisis incompleto')}**

### InterpretaciÃƒÂ³n

{'Ã¢Å“â€œ El sistema detecta correctamente que los datos anti-hologrÃƒÂ¡ficos NO producen holografÃƒÂ­a vÃƒÂ¡lida. Esto es evidencia de honestidad cientÃƒÂ­fica del pipeline.' if contract_check.get('status') == 'SUCCESS' else ''}
{'Ã¢Å¡Â  Pass rate moderado. Revisar quÃƒÂ© contratos pasaron y por quÃƒÂ©.' if contract_check.get('status') == 'WARNING' else ''}
{'Ã°Å¸Å¡Â¨ ALERTA: El sistema puede estar produciendo falsos positivos. InvestigaciÃƒÂ³n urgente necesaria.' if contract_check.get('status') == 'ALERT' else ''}

---

## 5. Archivos Generados

- Datos HDF5: `negative_control_{run_id}.h5`
- Este reporte: `negative_control_report_{run_id}.md`

---

*Generado automÃƒÂ¡ticamente por 04c_negative_controls.py*
*Proyecto CUERDAS-Maldacena*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Reporte generado: {report_path}")
    
    return report_path


def _format_list(items: List[str]) -> str:
    """Helper para formatear listas en markdown."""
    if not items or items == ['(ninguno)'] or items == ['(ninguna)']:
        return "- (ninguno)\n"
    return '\n'.join(f"- {item}" for item in items)


# ==============================================================================
# SECCIÃƒâ€œN 5: CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Control negativo para validaciÃƒÂ³n del pipeline CUERDAS-Maldacena',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Generar control negativo con parÃƒÂ¡metros por defecto
  python 04c_negative_controls.py --output_dir runs/negative_control
  
  # Especificar parÃƒÂ¡metros fÃƒÂ­sicos
  python 04c_negative_controls.py --mass 1.0 --lattice_size 100 --seed 42
  
  # Solo generar datos (sin ejecutar pipeline)
  python 04c_negative_controls.py --generate_only
"""
    )
    
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        default=Path('runs') / f'negative_control_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--mass', 
        type=float, 
        default=1.0,
        help='Masa del campo escalar (default: 1.0)'
    )
    parser.add_argument(
        '--lattice_size', 
        type=int, 
        default=100,
        help='TamaÃƒÂ±o del lattice (default: 100)'
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
        help='Nivel de ruido gaussiano (default: 0.01)'
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
    
    args = parser.parse_args()
    
    # Run ID ÃƒÂºnico
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.seed is not None:
        run_id += f"_seed{args.seed}"
    
    logger.info("="*60)
    logger.info("CONTROL NEGATIVO - CUERDAS-MALDACENA")
    logger.info("="*60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {args.output_dir}")
    
    # 1. Generar datos anti-hologrÃƒÂ¡ficos
    logger.info("\n[PASO 1] Generando datos anti-hologrÃƒÂ¡ficos...")
    data = generate_massive_scalar_flat_space(
        mass=args.mass,
        lattice_size=args.lattice_size,
        dim=args.dim,
        seed=args.seed,
        noise_level=args.noise
    )
    
    # 2. Guardar en formato pipeline
    logger.info("\n[PASO 2] Guardando datos en formato HDF5...")
    h5_path = save_negative_control_data(data, args.output_dir, run_id)
    
    if args.generate_only:
        logger.info("\n[COMPLETADO] Modo --generate_only: datos guardados, pipeline no ejecutado")
        return
    
    # 3. Ejecutar pipeline
    logger.info("\n[PASO 3] Ejecutando pipeline sobre control negativo...")
    results = run_pipeline_on_negative_control(h5_path, args.pipeline_dir)
    
    # 4. Verificar fallos esperados
    logger.info("\n[PASO 4] Verificando contratos...")
    contract_check = check_contracts_failure(results)
    
    # 5. Generar reporte
    logger.info("\n[PASO 5] Generando reporte...")
    report_path = generate_negative_control_report(
        data, results, contract_check, args.output_dir, run_id
    )
    
    # Resumen final
    logger.info("\n" + "="*60)
    logger.info("RESUMEN")
    logger.info("="*60)
    logger.info(f"Estado: {contract_check.get('status', 'INCOMPLETE')}")
    logger.info(f"Mensaje: {contract_check.get('message', 'Ver reporte')}")
    logger.info(f"Reporte: {report_path}")
    logger.info("="*60)
    
    # Exit code segÃƒÂºn resultado
    if contract_check.get('status') == 'SUCCESS':
        sys.exit(0)
    elif contract_check.get('status') == 'WARNING':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
