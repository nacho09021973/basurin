#!/usr/bin/env python3
# 06_build_bulk_eigenmodes_dataset.py
# CUERDAS – Bloque B: Espectro escalar (dataset de modos bulk)
#
# OBJETIVO
#   Recorrer las geometrías emergentes y construir un dataset honesto de modos bulk:
#     - Llamar a bulk_scalar_solver.py para cada sistema.
#     - Recopilar pares (Delta_UV, lambda_sl) con metadatos (familia, d, ...).
#     - Opcionalmente: Extraer Delta desde correladores de frontera.
#
# ENTRADAS
#   - runs/<experiment>/02_emergent_geometry_engine/geometry_emergent/*.h5
#   - Módulo bulk_scalar_solver.py (o bulk_scalar_solver_v2 si existe)
#   - Módulo boundary_delta_extractor.py (opcional, para extracción de Delta)
#
# SALIDAS (V3)
#   runs/<experiment>/06_build_bulk_eigenmodes_dataset/
#     bulk_modes_dataset.csv
#     bulk_modes_meta.json
#     stage_summary.json
#
# HONESTIDAD
#   - No se aplica ninguna fórmula teórica Delta(Delta-d).
#   - lambda_sl son autovalores Sturm—Liouville, NO masas holográficas por defecto.
#   - Delta extraído de correladores G2(x) ~ x^(-2Î”) es una MEDICIÓN, no teoría.
#   - El mapping mode_id â†’ operator se documenta en meta para auditoría.
#
# MIGRADO A V3: 2024-12-23

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# V3 INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
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
    from cuerdas_io import resolve_geometry_emergent_dir, update_run_manifest
    HAS_CUERDAS_IO = True
except ImportError:
    HAS_CUERDAS_IO = False

# Importar el solver v2 con nomenclatura honesta
try:
    import bulk_scalar_solver_v2 as bss  # type: ignore
except ImportError:
    try:
        import bulk_scalar_solver as bss  # type: ignore
    except ImportError:
        bss = None
        print("[WARN] bulk_scalar_solver no disponible.")

# Importar extractor de Delta desde correladores de frontera
try:
    from boundary_delta_extractor import (
        extract_deltas_from_hdf5, 
        get_delta_for_eigenmode, 
        get_extraction_metadata,
        DeltaExtraction
    )
    HAS_BOUNDARY_EXTRACTOR = True
except ImportError:
    HAS_BOUNDARY_EXTRACTOR = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Construir dataset de modos escalares (CSV canónico) a partir de geometrías emergentes. "
            "Nomenclatura honesta: lambda_sl (autovalores Sturm—Liouville)."
        )
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # V3: Argumentos estándar
    # ═══════════════════════════════════════════════════════════════════════════
    if HAS_STAGE_UTILS:
        add_standard_arguments(p)
    else:
        p.add_argument("--experiment", type=str, default=None)
        p.add_argument("--run-dir", type=str, default=None)
    
    # Legacy arguments (mantener compatibilidad)
    p.add_argument(
        "--geometry-dir",
        type=str,
        default=None,
        help="Directorio con .h5 de geometría emergente (legacy)",
    )

    # Salidas canónicas (legacy - en V3 se usan ctx.stage_dir)
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="CSV canónico (legacy, default: stage_dir/bulk_modes_dataset.csv)",
    )
    p.add_argument(
        "--output-meta",
        type=str,
        default=None,
        help="JSON meta del dataset (legacy)",
    )

    # Compat / usos auxiliares (Ising, etc.)
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="(Opcional) JSON agregador por sistema/family-d (compat/aux).",
    )

    p.add_argument(
        "--n-eigs",
        type=int,
        default=4,
        help="Número de autovalores/autovectores por geometría (default: 4)",
    )

    # Datasets dentro del HDF5: por defecto, layout emergent (02)
    p.add_argument(
        "--z-dataset",
        type=str,
        default="z_grid",
        help="Ruta al dataset z dentro del HDF5 (default: z_grid)",
    )
    p.add_argument(
        "--A-dataset",
        type=str,
        default="A_emergent",
        help="Ruta al dataset A(z) dentro del HDF5 (default: A_emergent)",
    )
    p.add_argument(
        "--f-dataset",
        type=str,
        default="f_emergent",
        help="Ruta al dataset f(z) dentro del HDF5 (default: f_emergent)",
    )
    
    # Control de extracción de Delta
    p.add_argument(
        "--delta-uv-source",
        type=str,
        choices=["solver", "boundary", "both"],
        default="solver",
        help=(
            "Fuente de Delta_UV: 'solver' (default, desde uv_exponents del solver), "
            "'boundary' (desde correladores G2 en boundary/), "
            "'both' (prioriza boundary, fallback a solver)"
        ),
    )

    return p.parse_args()


def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return x


def read_required_attrs(h5_path: Path) -> Dict[str, Any]:
    """Lee metadatos críticos. No hace defaults silenciosos."""
    with h5py.File(h5_path, "r") as f:
        # system_name / name
        system_name = _decode_if_bytes(f.attrs.get("system_name", f.attrs.get("name", h5_path.stem)))
        family = _decode_if_bytes(f.attrs.get("family", f.attrs.get("family_pred", None)))
        d = f.attrs.get("d", f.attrs.get("d_pred", None))
        z_dyn = f.attrs.get("z_dyn", np.nan)
        theta = f.attrs.get("theta", np.nan)

    if family is None:
        raise ValueError(f"[{h5_path.name}] Falta attrs 'family' (o 'family_pred')")
    if d is None:
        raise ValueError(f"[{h5_path.name}] Falta attrs 'd' (o 'd_pred')")
    try:
        d_int = int(d)
    except Exception as e:
        raise ValueError(f"[{h5_path.name}] Attr 'd' no es int: {d!r}") from e

    return {
        "system_name": str(system_name),
        "family": str(family),
        "d": d_int,
        "z_dyn": float(z_dyn) if z_dyn is not None else float("nan"),
        "theta": float(theta) if theta is not None else float("nan"),
    }


def pick_existing_dataset(h5_path: Path, candidates: Sequence[str]) -> str:
    """Devuelve el primer dataset existente en el HDF5."""
    with h5py.File(h5_path, "r") as f:
        for key in candidates:
            if key in f:
                return key
    raise KeyError(f"[{h5_path.name}] No existe ninguno de los datasets: {list(candidates)}")


@dataclass
class Row:
    system_name: str
    family: str
    d: int
    z_dyn: float
    theta: float
    mode_id: int
    lambda_sl: float
    Delta_UV: Optional[float]
    quality_flag: str
    is_ground_state: bool
    delta_source: str  # "boundary_correlator", "solver_uv", "none"


def write_csv(rows: List[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system_name",
        "family",
        "d",
        "z_dyn",
        "theta",
        "mode_id",
        "lambda_sl",
        "Delta_UV",
        "quality_flag",
        "is_ground_state",
        "delta_source",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "system_name": r.system_name,
                    "family": r.family,
                    "d": r.d,
                    "z_dyn": "" if np.isnan(r.z_dyn) else f"{r.z_dyn:.12g}",
                    "theta": "" if np.isnan(r.theta) else f"{r.theta:.12g}",
                    "mode_id": r.mode_id,
                    "lambda_sl": f"{r.lambda_sl:.16g}",
                    "Delta_UV": "" if r.Delta_UV is None else f"{r.Delta_UV:.16g}",
                    "quality_flag": r.quality_flag,
                    "is_ground_state": int(r.is_ground_state),
                    "delta_source": r.delta_source,
                }
            )


def build_legacy_json(rows: List[Row]) -> Dict[str, Any]:
    """Construye un JSON agregador compatible con loaders legacy (por-sistema y por family-d)."""
    systems: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.system_name not in systems:
            systems[r.system_name] = {
                "geometry_name": r.system_name,
                "family": r.family,
                "d": r.d,
                "n_modes": 0,
                "Delta_bulk_uv": [],
                "lambda_sl_bulk": [],
                "lambda_source": "bulk_eigenmode",
            }
        systems[r.system_name]["n_modes"] += 1
        systems[r.system_name]["lambda_sl_bulk"].append(float(r.lambda_sl))
        systems[r.system_name]["Delta_bulk_uv"].append(None if r.Delta_UV is None else float(r.Delta_UV))

    by_family_d: Dict[str, Dict[str, Any]] = {}
    for sys in systems.values():
        key = f"{sys['family']}_d{sys['d']}"
        if key not in by_family_d:
            by_family_d[key] = {
                "family": sys["family"],
                "d": sys["d"],
                "Delta_bulk_uv": [],
                "lambda_sl_bulk": [],
                "geometries": [],
            }
        by_family_d[key]["geometries"].append(sys["geometry_name"])
        by_family_d[key]["Delta_bulk_uv"].extend(sys["Delta_bulk_uv"])
        by_family_d[key]["lambda_sl_bulk"].extend(sys["lambda_sl_bulk"])

    return {
        "timestamp": datetime.now().isoformat(),
        "source": "06_build_bulk_eigenmodes_dataset",
        "nomenclature_version": "v2_lambda_sl",
        "systems": list(systems.values()),
        "by_family_d": by_family_d,
        "notes": [
            "Compat JSON: agregado por sistema/family-d.",
            "lambda_sl son autovalores Sturm—Liouville (NO masas holográficas por defecto).",
            "Delta_bulk_uv es el exponente UV estimado (de boundary o solver).",
        ],
    }


def main() -> int:
    args = parse_args()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # V3: Crear StageContext
    # ═══════════════════════════════════════════════════════════════════════════
    ctx = None
    if HAS_STAGE_UTILS:
        # Inferir experiment si no se proporciona
        if not getattr(args, 'experiment', None):
            args.experiment = infer_experiment(args)
        
        ctx = StageContext.from_args(
            args,
            stage_number="06",
            stage_slug="build_bulk_eigenmodes_dataset"
        )
        print(f"[V3] Experiment: {ctx.experiment}")
        print(f"[V3] Stage dir: {ctx.stage_dir}")
    
    # Resolver fuente de Delta
    use_boundary = args.delta_uv_source in ("boundary", "both")
    use_solver_delta = args.delta_uv_source in ("solver", "both")
    boundary_priority = args.delta_uv_source == "both"  # priorizar boundary si ambos
    
    # Validar disponibilidad
    if use_boundary and not HAS_BOUNDARY_EXTRACTOR:
        print("[ERROR] --delta-uv-source boundary/both requiere boundary_delta_extractor.py")
        print("        Copiar el módulo al directorio del proyecto.")
        if ctx:
            ctx.write_summary(status="ERROR", counts={"error": "boundary_extractor_missing"})
        return 2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESOLVER INPUT (geometry_dir)
    # ═══════════════════════════════════════════════════════════════════════════
    geom_dir = None
    
    # Prioridad 1: --geometry-dir explícito
    if args.geometry_dir:
        geom_dir = Path(args.geometry_dir).resolve()
    
    # Prioridad 2: V3 - buscar en 02_emergent_geometry_engine/geometry_emergent
    if geom_dir is None and ctx:
        candidate = ctx.run_root / "02_emergent_geometry_engine" / "geometry_emergent"
        if candidate.exists() and list(candidate.glob("*.h5")):
            geom_dir = candidate
            print(f"[V3] Geometry dir desde stage 02: {geom_dir}")
    
    # Prioridad 2b: Fallback a 01_generate_sandbox_geometries (sandbox original)
    if geom_dir is None and ctx:
        candidate = ctx.run_root / "01_generate_sandbox_geometries"
        if candidate.exists() and list(candidate.glob("*.h5")):
            geom_dir = candidate
            print(f"[V3] Geometry dir desde stage 01 (sandbox): {geom_dir}")
    
    # Prioridad 3: --run-dir legacy con cuerdas_io
    if geom_dir is None and args.run_dir and HAS_CUERDAS_IO:
        run_dir = Path(args.run_dir).resolve()
        geom_dir = resolve_geometry_emergent_dir(run_dir=run_dir)
    
    if geom_dir is None:
        print("[ERROR] Debe proporcionar --experiment, --run-dir o --geometry-dir")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "no_geometry_dir"})
        return 2
    
    if not geom_dir.exists() or not geom_dir.is_dir():
        print(f"[ERROR] --geometry-dir no es un directorio válido: {geom_dir}")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "geometry_dir_not_found"})
        return 2

    h5_files = sorted(geom_dir.glob("*.h5"))
    if not h5_files:
        print(f"[ERROR] No se encontraron .h5 en {geom_dir}")
        if ctx:
            ctx.write_summary(status="INCOMPLETE", counts={"error": "no_h5_files"})
        return 2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESOLVER OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════════
    if ctx:
        out_csv = ctx.stage_dir / "bulk_modes_dataset.csv"
        out_meta = ctx.stage_dir / "bulk_modes_meta.json"
    elif args.output_csv:
        out_csv = Path(args.output_csv).resolve()
        out_meta = Path(args.output_meta).resolve() if args.output_meta else out_csv.with_name("bulk_modes_meta.json")
    elif args.run_dir:
        out_dir = Path(args.run_dir).resolve() / "bulk_eigenmodes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "bulk_modes_dataset.csv"
        out_meta = out_dir / "bulk_modes_meta.json"
    else:
        out_csv = Path("runs/bulk_eigenmodes/bulk_modes_dataset.csv").resolve()
        out_meta = Path("runs/bulk_eigenmodes/bulk_modes_meta.json").resolve()
    
    out_json = Path(args.output_json).resolve() if args.output_json else None

    # ═══════════════════════════════════════════════════════════════════════════
    # BANNER
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("DATASET BULK EIGENMODES (CSV canónico) – CUERDAS Bloque B")
    print("Nomenclatura honesta: lambda_sl (autovalores Sturm—Liouville)")
    print("=" * 70)
    print(f"Geometrías desde: {geom_dir}")
    print(f"N_max modos por geometría: {args.n_eigs}")
    print(f"Salida CSV:  {out_csv}")
    print(f"Salida META: {out_meta}")
    if out_json:
        print(f"Salida JSON (compat): {out_json}")
    print(f"Fuente Delta_UV: {args.delta_uv_source}")
    if use_boundary:
        print(f"  boundary_delta_extractor: {'disponible' if HAS_BOUNDARY_EXTRACTOR else 'NO disponible'}")
    print("=" * 70)

    rows: List[Row] = []
    failed: List[Dict[str, Any]] = []
    compat_used_keys: Set[str] = set()
    
    # Estadísticas de fuentes de Delta
    stats = {
        "boundary_extractions": 0,
        "solver_extractions": 0,
        "no_delta": 0,
    }
    
    # Metadata de boundary extractions (para auditoría)
    boundary_metadata_by_system: Dict[str, Any] = {}

    for h5_path in h5_files:
        try:
            meta = read_required_attrs(h5_path)
        except Exception as e:
            failed.append({"file": str(h5_path), "stage": "read_attrs", "error": str(e)})
            print(f"\n>> [WARN] Fallo leyendo attrs de {h5_path.name}: {e}")
            continue
            
        system_name = meta["system_name"]
        family = meta["family"]
        d = meta["d"]
        z_dyn = meta["z_dyn"]
        theta = meta["theta"]

        print(f"\n>> Procesando: {system_name} (family={family}, d={d})")

        # === PASO 1: Extraer Delta desde correladores de frontera (si corresponde) ===
        boundary_deltas: Dict[str, Any] = {}
        if use_boundary and HAS_BOUNDARY_EXTRACTOR:
            try:
                boundary_deltas = extract_deltas_from_hdf5(h5_path)
                if boundary_deltas:
                    ops = ", ".join(f"{k}(Δ={v.Delta:.4f})" for k, v in 
                                    sorted(boundary_deltas.items(), key=lambda x: x[1].Delta))
                    print(f"   Deltas de boundary: {ops}")
                    # Guardar metadata para auditoría
                    boundary_metadata_by_system[system_name] = get_extraction_metadata(boundary_deltas)
            except Exception as e:
                print(f"   [WARN] Fallo extracción boundary: {e}")

        # === PASO 2: Llamar al solver ===
        spec = None
        lambda_list = []
        Delta_uv_list_solver = []
        
        if bss is not None:
            # Resolver datasets para el solver
            try:
                z_ds = pick_existing_dataset(h5_path, [args.z_dataset, "z_grid", "bulk_truth/z_grid"])
                A_ds = pick_existing_dataset(h5_path, [args.A_dataset, "A_emergent", "A_of_z", "bulk_truth/A_truth"])
                f_ds = pick_existing_dataset(h5_path, [args.f_dataset, "f_emergent", "f_of_z", "bulk_truth/f_truth"])
                print(f"   datasets: z={z_ds}, A={A_ds}, f={f_ds}")
            except Exception as e:
                failed.append({"system_name": system_name, "file": str(h5_path), "stage": "dataset_resolve", "error": str(e)})
                print(f"   [WARN] no se pudieron resolver datasets: {e}")
                if not boundary_deltas:
                    continue

            try:
                spec = bss.solve_geometry(
                    h5_path=h5_path,
                    n_eigs=args.n_eigs,
                    z_dataset=z_ds,
                    A_dataset=A_ds,
                    f_dataset=f_ds,
                )
                
                # Compatibilidad: lambda_sl (nuevo) o m2L2* (legacy)
                if "lambda_sl" in spec:
                    lambda_list = spec["lambda_sl"]
                    used_key = "lambda_sl"
                elif "m2L2" in spec:
                    lambda_list = spec["m2L2"]
                    used_key = "m2L2"
                elif "m2L2_legacy" in spec:
                    lambda_list = spec["m2L2_legacy"]
                    used_key = "m2L2_legacy"
                else:
                    lambda_list = []
                    used_key = "none"
                
                if used_key != "none":
                    compat_used_keys.add(used_key)
                
                Delta_uv_list_solver = spec.get("uv_exponents", [])
                
            except Exception as e:
                failed.append({"system_name": system_name, "file": str(h5_path), "stage": "solver", "error": str(e)})
                print(f"   [WARN] Fallo solver: {e}")

        # === PASO 3: Construir filas del CSV ===
        n_modes = len(lambda_list) if lambda_list else 0
        
        if n_modes == 0:
            print(f"   [WARN] Sin modos lambda_sl; se omite sistema.")
            continue

        n_added = 0
        for mode_id in range(n_modes):
            # Obtener lambda_sl
            lam = lambda_list[mode_id]
            try:
                lam_f = float(lam)
            except Exception:
                continue
            if not np.isfinite(lam_f) or lam_f <= 0:
                continue

            # === Determinar Delta_UV según --delta-uv-source ===
            Delta_uv: Optional[float] = None
            quality = "ok"
            delta_source = "none"
            
            # Opción 1: boundary (o both con prioridad boundary)
            if use_boundary and boundary_deltas:
                bnd_delta, bnd_quality = get_delta_for_eigenmode(boundary_deltas, mode_id)
                if bnd_delta is not None:
                    Delta_uv = bnd_delta
                    quality = bnd_quality
                    delta_source = "boundary_correlator"
                    stats["boundary_extractions"] += 1
            
            # Opción 2: solver (o fallback si boundary no dio resultado)
            if Delta_uv is None and use_solver_delta and mode_id < len(Delta_uv_list_solver):
                dv = Delta_uv_list_solver[mode_id]
                if dv is not None and isinstance(dv, (int, float)) and np.isfinite(float(dv)):
                    try:
                        Delta_uv = float(dv)
                        quality = "ok"
                        delta_source = "solver_uv"
                        stats["solver_extractions"] += 1
                    except Exception:
                        pass
            
            # Sin Delta disponible
            if Delta_uv is None:
                quality = "uv_missing"
                delta_source = "none"
                stats["no_delta"] += 1

            rows.append(
                Row(
                    system_name=system_name,
                    family=family,
                    d=d,
                    z_dyn=z_dyn,
                    theta=theta,
                    mode_id=int(mode_id),
                    lambda_sl=lam_f,
                    Delta_UV=Delta_uv,
                    quality_flag=quality,
                    is_ground_state=(mode_id == 0),
                    delta_source=delta_source,
                )
            )
            n_added += 1

        if n_added == 0:
            print("   [WARN] Sin modos válidos; se omite sistema.")
        else:
            # Mostrar primer modo
            ex = next((r for r in rows if r.system_name == system_name), None)
            if ex is not None:
                dv = "(vacío)" if ex.Delta_UV is None else f"{ex.Delta_UV:.6f}"
                print(f"   OK: {n_added} filas. Ejemplo: λ_sl={ex.lambda_sl:.6f}, Î”={dv} (src={ex.delta_source})")

    # ═══════════════════════════════════════════════════════════════════════════
    # ESCRIBIR OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════════
    write_csv(rows, out_csv)

    meta_out = {
        "timestamp": datetime.now().isoformat(),
        "nomenclature_version": "v2_lambda_sl",
        "geometry_dir": str(geom_dir),
        "n_geometries_scanned": len(h5_files),
        "n_geometries_solved": len(sorted(set(r.system_name for r in rows))),
        "n_rows": len(rows),
        "n_eigs": int(args.n_eigs),
        "datasets_requested": {"z": args.z_dataset, "A": args.A_dataset, "f": args.f_dataset},
        "solver_module": getattr(bss, "__name__", "bulk_scalar_solver") if bss else "none",
        "compat_used_keys": sorted(list(compat_used_keys)),
        "all_v2_clean": compat_used_keys == {"lambda_sl"} or len(compat_used_keys) == 0,
        "failed_systems": failed,
        "delta_extraction": {
            "delta_uv_source": args.delta_uv_source,
            "boundary_extractor_available": HAS_BOUNDARY_EXTRACTOR,
            "stats": stats,
        },
        "boundary_extraction_metadata": boundary_metadata_by_system,
        "notes": [
            "CSV canónico para 07: system_name,family,d,mode_id,lambda_sl,Delta_UV (+ opcionales).",
            "lambda_sl son autovalores Sturm—Liouville (NO masas por defecto).",
            "delta_source indica origen: boundary_correlator, solver_uv, o none.",
            "boundary_extraction_metadata contiene el mapping mode_id â†’ operator para auditoría.",
        ],
    }

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(build_legacy_json(rows), indent=2, ensure_ascii=False))

    # ═══════════════════════════════════════════════════════════════════════════
    # RESUMEN
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[OK] Dataset bulk-eigenmodes generado")
    print(f"  CSV :  {out_csv}")
    print(f"  META:  {out_meta}")
    if out_json:
        print(f"  JSON:  {out_json}")
    print(f"\n  ESTADÍSTICAS DE DELTA (fuente: {args.delta_uv_source}):")
    print(f"    - Desde boundary correlators: {stats['boundary_extractions']}")
    print(f"    - Desde solver UV:            {stats['solver_extractions']}")
    print(f"    - Sin Delta disponible:       {stats['no_delta']}")
    if boundary_metadata_by_system:
        print(f"\n  AUDITORÍA: mapping mode_id → operator guardado en bulk_modes_meta.json")
    if failed:
        print(f"\n  WARN: {len(failed)} sistemas fallaron (ver bulk_modes_meta.json)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # V3: Registrar artefactos y escribir summary
    # ═══════════════════════════════════════════════════════════════════════════
    if ctx:
        ctx.record_artifact("bulk_modes_csv", out_csv)
        ctx.record_artifact("bulk_modes_meta", out_meta)
        ctx.record_artifact("geometry_dir_input", geom_dir)
        if out_json:
            ctx.record_artifact("bulk_modes_json_legacy", out_json)
        
        status = "OK" if len(rows) > 0 else "WARNING"
        if len(failed) > len(h5_files) // 2:
            status = "WARNING"
        
        ctx.write_summary(
            status=status,
            counts={
                "geometries_scanned": len(h5_files),
                "geometries_solved": len(set(r.system_name for r in rows)),
                "rows_generated": len(rows),
                "failed_systems": len(failed),
                "boundary_extractions": stats["boundary_extractions"],
                "solver_extractions": stats["solver_extractions"],
                "no_delta": stats["no_delta"],
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
                    "bulk_eigenmodes_dir": str(out_csv.parent.relative_to(run_dir)
                                               if out_csv.parent.is_relative_to(run_dir)
                                               else out_csv.parent),
                    "bulk_modes_csv": str(out_csv.relative_to(run_dir)
                                          if out_csv.is_relative_to(run_dir)
                                          else out_csv),
                    "bulk_modes_meta": str(out_meta.relative_to(run_dir)
                                           if out_meta.is_relative_to(run_dir)
                                           else out_meta),
                }
            )
            print(f"  Manifest actualizado (legacy)")
        except Exception as e:
            print(f"  [WARN] No se pudo actualizar manifest: {e}")
    
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
