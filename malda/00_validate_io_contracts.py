#!/usr/bin/env python3
"""
00_validate_io_contracts.py — Validador de Contratos IO v1 para CUERDAS-Maldacena

Este script valida que los archivos en runs/ cumplan con el contrato IO v1
definido en IO_CONTRACTS_V1.md.

Uso:
    python 00_validate_io_contracts.py --runs-dir runs/
    python 00_validate_io_contracts.py --runs-dir runs/ --output report.json --strict

El validador produce:
    - Reporte JSON con PASS/FAIL por archivo y razones
    - Exit code 0 si todo pasa, 1 si hay fallos (en modo --strict)

Autor: CUERDAS-Maldacena Team
Versión: 1.0 (2025-12)
Contrato: IO_CONTRACTS_V1.md
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd


# =============================================================================
# Data classes para el reporte
# =============================================================================

@dataclass
class ValidationIssue:
    """Un problema encontrado durante la validación."""
    level: str  # "ERROR", "WARN", "INFO"
    code: str   # Código único del problema
    message: str
    field: Optional[str] = None


@dataclass
class FileValidation:
    """Resultado de validación de un archivo."""
    filepath: str
    filetype: str  # "sandbox_h5", "emergent_h5", "modes_csv", "json"
    status: str    # "PASS", "FAIL", "WARN"
    issues: list = field(default_factory=list)
    metadata_extracted: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Reporte completo de validación."""
    timestamp: str
    contract_version: str
    runs_dir: str
    total_files: int
    passed: int
    failed: int
    warnings: int
    files: list = field(default_factory=list)


# =============================================================================
# Constantes del contrato
# =============================================================================

VALID_FAMILIES = {"ads", "lifshitz", "hvlf", "hyperscaling", "deformed", "unknown", 
                  "ising3d", "real", "ising_3d", "dpbrane"}
VALID_CATEGORIES = {"known", "test", "unknown"}
VALID_PROVENANCES = {"train", "inference", "sandbox", "emergent"}

CANONICAL_DATASETS = {"z_grid", "A_of_z", "f_of_z"}
OPTIONAL_DATASETS = {"R_of_z", "phi_of_z"}
LEGACY_ALIASES = {
    "A_emergent": "A_of_z",
    "f_emergent": "f_of_z", 
    "R_emergent": "R_of_z"
}

BULK_TRUTH_DATASETS = {"z_grid", "A_truth", "f_truth"}

MODES_CSV_REQUIRED_COLS = {"system_name", "family", "d", "mode_id", "lambda_sl", "Delta_UV"}


# =============================================================================
# Funciones de validación
# =============================================================================

def extract_d_from_name(name: str) -> Optional[int]:
    """Extrae dimensión d del nombre si contiene _d<k>_."""
    match = re.search(r"_d(\d+)_", name)
    if match:
        return int(match.group(1))
    return None


def validate_monotonic_increasing(arr: np.ndarray, name: str) -> list:
    """Valida que un array sea estrictamente creciente."""
    issues = []
    if len(arr) < 2:
        return issues
    
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        n_violations = np.sum(diffs <= 0)
        issues.append(ValidationIssue(
            level="ERROR",
            code="MONOTONIC_VIOLATION",
            message=f"{name} no es estrictamente creciente ({n_violations} violaciones)",
            field=name
        ))
    return issues


def validate_consistent_lengths(datasets: dict, reference: str = "z_grid") -> list:
    """Valida que todos los datasets tengan la misma longitud que el de referencia."""
    issues = []
    if reference not in datasets:
        return issues
    
    ref_len = len(datasets[reference])
    for name, arr in datasets.items():
        if name != reference and len(arr) != ref_len:
            issues.append(ValidationIssue(
                level="ERROR",
                code="LENGTH_MISMATCH",
                message=f"{name} tiene longitud {len(arr)}, esperada {ref_len} (como {reference})",
                field=name
            ))
    return issues


def validate_sandbox_h5(filepath: str) -> FileValidation:
    """Valida un archivo HDF5 de sandbox."""
    result = FileValidation(
        filepath=filepath,
        filetype="sandbox_h5",
        status="PASS"
    )
    
    try:
        with h5py.File(filepath, "r") as f:
            # --- Atributos raíz obligatorios ---
            root_attrs = dict(f.attrs)
            result.metadata_extracted["root_attrs"] = {k: str(v) for k, v in root_attrs.items()}
            
            # name o system_name
            name = root_attrs.get("name") or root_attrs.get("system_name")
            if not name:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_ATTR",
                    message="Falta atributo 'name' o 'system_name' en raíz",
                    field="name"
                ))
            
            # family
            family = root_attrs.get("family")
            if not family:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_ATTR",
                    message="Falta atributo 'family' en raíz",
                    field="family"
                ))
            elif str(family).lower() not in VALID_FAMILIES:
                result.issues.append(ValidationIssue(
                    level="WARN",
                    code="UNKNOWN_FAMILY",
                    message=f"Familia '{family}' no está en la lista conocida",
                    field="family"
                ))
            
            # d
            d = root_attrs.get("d")
            if d is None:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_ATTR",
                    message="Falta atributo 'd' en raíz",
                    field="d"
                ))
            else:
                # Verificar consistencia con nombre
                filename = os.path.basename(filepath)
                d_from_name = extract_d_from_name(filename)
                if d_from_name is not None and int(d) != d_from_name:
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="D_INCONSISTENT",
                        message=f"Atributo d={d} no coincide con nombre (d={d_from_name})",
                        field="d"
                    ))
            # category (obligatorio)
            category = root_attrs.get("category")
            if not category:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_ATTR",
                    message="Falta atributo 'category' en raíz",
                    field="category"
                ))
            elif str(category).lower() not in VALID_CATEGORIES:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="UNKNOWN_CATEGORY",
                    message=f"Categoría '{category}' no está en la lista conocida",
                    field="category"
                ))
            
            # --- Grupo boundary/ ---
            if "boundary" not in f:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_GROUP",
                    message="No existe grupo 'boundary/' (obligatorio en sandbox)",
                    field="boundary"
                ))
            else:
                boundary = f["boundary"]
                boundary_attrs = dict(boundary.attrs)
                
                # Verificar attrs obligatorios en boundary
                boundary_d = boundary_attrs.get("d")
                if boundary_d is None:
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="MISSING_ATTR",
                        message="Falta boundary.attrs['d']",
                        field="boundary/d"
                    ))
                elif d is not None and int(boundary_d) != int(d):
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="D_INCONSISTENT",
                        message=f"boundary.attrs['d']={boundary_d} != root.attrs['d']={d}",
                        field="boundary/d"
                    ))

                boundary_family = boundary_attrs.get("family")
                if boundary_family is None or str(boundary_family).strip() == "":
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="MISSING_ATTR",
                        message="Falta boundary.attrs['family']",
                        field="boundary/family"
                    ))
                elif family is not None and str(boundary_family).lower() != str(family).lower():
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="FAMILY_INCONSISTENT",
                        message=f"boundary.attrs['family']={boundary_family} != root.attrs['family']={family}",
                        field="boundary/family"
                    ))
            
            # --- Grupo bulk_truth/ ---
            if "bulk_truth" not in f:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_GROUP",
                    message="No existe grupo 'bulk_truth/' (obligatorio en sandbox)",
                    field="bulk_truth"
                ))
            else:
                bulk_truth = f["bulk_truth"]
                datasets = {}
                
                for ds_name in BULK_TRUTH_DATASETS:
                    if ds_name not in bulk_truth:
                        result.issues.append(ValidationIssue(
                            level="ERROR",
                            code="MISSING_DATASET",
                            message=f"Falta dataset 'bulk_truth/{ds_name}'",
                            field=f"bulk_truth/{ds_name}"
                        ))
                    else:
                        datasets[ds_name] = bulk_truth[ds_name][:]
                
                # Validar monotonicidad de z_grid
                if "z_grid" in datasets:
                    result.issues.extend(validate_monotonic_increasing(
                        datasets["z_grid"], "bulk_truth/z_grid"
                    ))
                
                # Validar longitudes consistentes
                result.issues.extend(validate_consistent_lengths(datasets))
    
    except Exception as e:
        result.issues.append(ValidationIssue(
            level="ERROR",
            code="READ_ERROR",
            message=f"Error leyendo archivo: {str(e)}"
        ))
    
    # Determinar status final
    has_errors = any(i.level == "ERROR" for i in result.issues)
    has_warnings = any(i.level == "WARN" for i in result.issues)
    
    if has_errors:
        result.status = "FAIL"
    elif has_warnings:
        result.status = "WARN"
    else:
        result.status = "PASS"
    
    return result


def validate_emergent_h5(filepath: str) -> FileValidation:
    """Valida un archivo HDF5 de geometría emergente."""
    result = FileValidation(
        filepath=filepath,
        filetype="emergent_h5",
        status="PASS"
    )

    try:
        with h5py.File(filepath, "r") as f:
            root_attrs = dict(f.attrs)
            result.metadata_extracted["root_attrs"] = {k: str(v) for k, v in root_attrs.items()}

            # --- Atributos obligatorios ---
            system_name = root_attrs.get("system_name") or root_attrs.get("name")
            if not system_name:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_ATTR",
                    message="Falta atributo 'system_name' o 'name'",
                    field="system_name"
                ))

            # family: canónico. family_pred es solo auxiliar.
            family = root_attrs.get("family")
            if not family:
                if root_attrs.get("family_pred"):
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="MISSING_CANONICAL_ATTR",
                        message="Falta 'family'. No se acepta solo 'family_pred' según IO_CONTRACTS_V1.",
                        field="family"
                    ))
                else:
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="MISSING_ATTR",
                        message="Falta atributo 'family'",
                        field="family"
                    ))

            # d: canónico. d_pred es solo auxiliar.
            d = root_attrs.get("d")
            if d is None:
                if root_attrs.get("d_pred") is not None:
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="MISSING_CANONICAL_ATTR",
                        message="Falta atributo 'd' (solo existe 'd_pred')",
                        field="d"
                    ))
                else:
                    result.issues.append(ValidationIssue(
                        level="ERROR",
                        code="MISSING_ATTR",
                        message="Falta atributo 'd'",
                        field="d"
                    ))

            # provenance: obligatorio
            provenance = root_attrs.get("provenance")
            if not provenance:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_ATTR",
                    message="Falta atributo 'provenance' (obligatorio en geometría emergente)",
                    field="provenance"
                ))

            # --- Datasets canónicos ---
            datasets = {}

            # z_grid (obligatorio)
            if "z_grid" not in f:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_DATASET",
                    message="Falta dataset 'z_grid'",
                    field="z_grid"
                ))
            else:
                datasets["z_grid"] = f["z_grid"][:]

            # A_of_z (obligatorio; fallback legacy)
            if "A_of_z" in f:
                datasets["A_of_z"] = f["A_of_z"][:]
            elif "A_emergent" in f:
                datasets["A_of_z"] = f["A_emergent"][:]
                result.issues.append(ValidationIssue(
                    level="WARN",
                    code="LEGACY_DATASET",
                    message="Usando alias legacy 'A_emergent' (debería ser 'A_of_z')",
                    field="A_emergent"
                ))
            else:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_DATASET",
                    message="Falta dataset 'A_of_z' (y no hay fallback 'A_emergent')",
                    field="A_of_z"
                ))

            # f_of_z (obligatorio; fallback legacy)
            if "f_of_z" in f:
                datasets["f_of_z"] = f["f_of_z"][:]
            elif "f_emergent" in f:
                datasets["f_of_z"] = f["f_emergent"][:]
                result.issues.append(ValidationIssue(
                    level="WARN",
                    code="LEGACY_DATASET",
                    message="Usando alias legacy 'f_emergent' (debería ser 'f_of_z')",
                    field="f_emergent"
                ))
            else:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_DATASET",
                    message="Falta dataset 'f_of_z' (y no hay fallback 'f_emergent')",
                    field="f_of_z"
                ))

            # R_of_z (opcional; fallback legacy)
            if "R_of_z" in f:
                datasets["R_of_z"] = f["R_of_z"][:]
            elif "R_emergent" in f:
                datasets["R_of_z"] = f["R_emergent"][:]
                result.issues.append(ValidationIssue(
                    level="INFO",
                    code="LEGACY_DATASET",
                    message="Usando alias legacy 'R_emergent' (debería ser 'R_of_z')",
                    field="R_emergent"
                ))

            # --- Reglas de integridad ---
            if "z_grid" in datasets:
                result.issues.extend(validate_monotonic_increasing(datasets["z_grid"], "z_grid"))

            result.issues.extend(validate_consistent_lengths(datasets))

            # Si provenance indica inferencia, no debe existir bulk_truth
            if provenance and "inference" in str(provenance).lower() and "bulk_truth" in f:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="UNEXPECTED_GROUP",
                    message="Archivo con provenance de inferencia tiene grupo 'bulk_truth/'",
                    field="bulk_truth"
                ))

    except Exception as e:
        result.issues.append(ValidationIssue(
            level="ERROR",
            code="READ_ERROR",
            message=f"Error leyendo archivo: {str(e)}"
        ))

    # Determinar status final
    has_errors = any(i.level == "ERROR" for i in result.issues)
    has_warnings = any(i.level == "WARN" for i in result.issues)

    if has_errors:
        result.status = "FAIL"
    elif has_warnings:
        result.status = "WARN"
    else:
        result.status = "PASS"

    return result


def validate_modes_csv(filepath: str) -> FileValidation:
    """Valida un archivo CSV de modos bulk."""
    result = FileValidation(
        filepath=filepath,
        filetype="modes_csv",
        status="PASS"
    )
    
    try:
        df = pd.read_csv(filepath)
        result.metadata_extracted["n_rows"] = len(df)
        result.metadata_extracted["columns"] = list(df.columns)
        
        # Verificar columnas obligatorias
        missing_cols = MODES_CSV_REQUIRED_COLS - set(df.columns)
        if missing_cols:
            result.issues.append(ValidationIssue(
                level="ERROR",
                code="MISSING_COLUMNS",
                message=f"Faltan columnas obligatorias: {missing_cols}",
                field="columns"
            ))
        
        # Verificar que no exista m2L2 como columna principal
        if "m2L2" in df.columns and "lambda_sl" not in df.columns:
            result.issues.append(ValidationIssue(
                level="ERROR",
                code="WRONG_NOMENCLATURE",
                message="Columna 'm2L2' sin 'lambda_sl'. Usar 'lambda_sl' como canónico.",
                field="m2L2"
            ))
        
        # Verificar valores válidos
        if "lambda_sl" in df.columns:
            n_nan = df["lambda_sl"].isna().sum()
            n_inf = np.isinf(df["lambda_sl"]).sum()
            if n_nan > 0:
                result.issues.append(ValidationIssue(
                    level="WARN",
                    code="INVALID_VALUES",
                    message=f"lambda_sl tiene {n_nan} valores NaN",
                    field="lambda_sl"
                ))
            if n_inf > 0:
                result.issues.append(ValidationIssue(
                    level="WARN",
                    code="INVALID_VALUES",
                    message=f"lambda_sl tiene {n_inf} valores Inf",
                    field="lambda_sl"
                ))
        
        if "Delta_UV" in df.columns:
            n_nan = df["Delta_UV"].isna().sum()
            if n_nan > 0:
                result.issues.append(ValidationIssue(
                    level="WARN",
                    code="INVALID_VALUES",
                    message=f"Delta_UV tiene {n_nan} valores NaN",
                    field="Delta_UV"
                ))
    
    except Exception as e:
        result.issues.append(ValidationIssue(
            level="ERROR",
            code="READ_ERROR",
            message=f"Error leyendo CSV: {str(e)}"
        ))
    
    # Determinar status final
    has_errors = any(i.level == "ERROR" for i in result.issues)
    has_warnings = any(i.level == "WARN" for i in result.issues)
    
    if has_errors:
        result.status = "FAIL"
    elif has_warnings:
        result.status = "WARN"
    else:
        result.status = "PASS"
    
    return result


def validate_dictionary_json(filepath: str) -> FileValidation:
    """Valida un archivo JSON de diccionario lambda_sl."""
    result = FileValidation(
        filepath=filepath,
        filetype="dictionary_json",
        status="PASS"
    )
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        result.metadata_extracted["keys"] = list(data.keys())
        
        # Verificar campos obligatorios
        required_fields = ["config", "discovery_results", "data_stats"]
        for field in required_fields:
            if field not in data:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_FIELD",
                    message=f"Falta campo obligatorio '{field}'",
                    field=field
                ))
        
        # Verificar discovery_results
        if "discovery_results" in data:
            dr = data["discovery_results"]
            if "best_equation" not in dr:
                result.issues.append(ValidationIssue(
                    level="ERROR",
                    code="MISSING_FIELD",
                    message="Falta 'discovery_results.best_equation'",
                    field="discovery_results.best_equation"
                ))
            if "test_metrics" not in dr:
                result.issues.append(ValidationIssue(
                    level="WARN",
                    code="MISSING_FIELD",
                    message="Falta 'discovery_results.test_metrics'",
                    field="discovery_results.test_metrics"
                ))
        
        # Verificar nomenclature_version
        if "nomenclature_version" not in data:
            result.issues.append(ValidationIssue(
                level="WARN",
                code="MISSING_FIELD",
                message="Falta 'nomenclature_version' (recomendado: 'v2_lambda_sl')",
                field="nomenclature_version"
            ))
    
    except json.JSONDecodeError as e:
        result.issues.append(ValidationIssue(
            level="ERROR",
            code="PARSE_ERROR",
            message=f"Error parseando JSON: {str(e)}"
        ))
    except Exception as e:
        result.issues.append(ValidationIssue(
            level="ERROR",
            code="READ_ERROR",
            message=f"Error leyendo archivo: {str(e)}"
        ))
    
    # Determinar status final
    has_errors = any(i.level == "ERROR" for i in result.issues)
    has_warnings = any(i.level == "WARN" for i in result.issues)
    
    if has_errors:
        result.status = "FAIL"
    elif has_warnings:
        result.status = "WARN"
    else:
        result.status = "PASS"
    
    return result


# =============================================================================
# Descubrimiento de archivos
# =============================================================================

def discover_files(runs_dir: str) -> dict:
    """Descubre archivos a validar en runs_dir."""
    files = {
        "sandbox_h5": [],
        "emergent_h5": [],
        "modes_csv": [],
        "dictionary_json": [],
        "atlas_json": [],
        "fase12_json": []
    }
    
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return files
    
    # Sandbox HDF5
    for pattern in ["sandbox_geometries/*.h5", "sandbox_geometries/**/*.h5"]:
        for p in runs_path.glob(pattern):
            if p.is_file() and str(p) not in files["sandbox_h5"]:
                files["sandbox_h5"].append(str(p))
    
    # Emergent geometry HDF5
    for pattern in ["emergent_geometry/geometry_emergent/*.h5", 
                    "**/geometry_emergent/*.h5",
                    "emergent_geometry/*.h5"]:
        for p in runs_path.glob(pattern):
            if p.is_file() and str(p) not in files["emergent_h5"]:
                # Excluir si ya está en sandbox
                if str(p) not in files["sandbox_h5"]:
                    files["emergent_h5"].append(str(p))
    
    # Modes CSV
    for pattern in ["bulk_eigenmodes/*.csv", "**/bulk_modes*.csv"]:
        for p in runs_path.glob(pattern):
            if p.is_file():
                files["modes_csv"].append(str(p))
    
    # Dictionary JSON
    for pattern in ["emergent_dictionary/*dictionary*.json", 
                    "**/lambda_sl_dictionary*.json"]:
        for p in runs_path.glob(pattern):
            if p.is_file():
                files["dictionary_json"].append(str(p))
    
    # Atlas JSON
    for pattern in ["holographic_dictionary/*.json"]:
        for p in runs_path.glob(pattern):
            if p.is_file():
                files["atlas_json"].append(str(p))
    
    # Fase12 JSON
    for pattern in ["**/fase12_report*.json", "**/fase12/predictions/*.json"]:
        for p in runs_path.glob(pattern):
            if p.is_file():
                files["fase12_json"].append(str(p))
    
    return files


# =============================================================================
# Main
# =============================================================================

def run_validation(runs_dir: str, verbose: bool = True) -> ValidationReport:
    """Ejecuta validación completa."""
    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        contract_version="v1.0",
        runs_dir=runs_dir,
        total_files=0,
        passed=0,
        failed=0,
        warnings=0
    )
    
    files = discover_files(runs_dir)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CUERDAS-Maldacena IO Contract Validator v1.0")
        print(f"{'='*60}")
        print(f"Directorio: {runs_dir}")
        print(f"Timestamp: {report.timestamp}")
        print()
    
    # Validar sandbox HDF5
    if verbose and files["sandbox_h5"]:
        print(f"\n--- Sandbox HDF5 ({len(files['sandbox_h5'])} archivos) ---")
    for filepath in files["sandbox_h5"]:
        result = validate_sandbox_h5(filepath)
        report.files.append(result)
        report.total_files += 1
        if result.status == "PASS":
            report.passed += 1
        elif result.status == "FAIL":
            report.failed += 1
        else:
            report.warnings += 1
        
        if verbose:
            status_icon = "✓" if result.status == "PASS" else ("✗" if result.status == "FAIL" else "⚠")
            print(f"  {status_icon} {os.path.basename(filepath)}: {result.status}")
            for issue in result.issues:
                if issue.level in ["ERROR", "WARN"]:
                    print(f"      [{issue.level}] {issue.message}")
    
    # Validar emergent HDF5
    if verbose and files["emergent_h5"]:
        print(f"\n--- Emergent HDF5 ({len(files['emergent_h5'])} archivos) ---")
    for filepath in files["emergent_h5"]:
        result = validate_emergent_h5(filepath)
        report.files.append(result)
        report.total_files += 1
        if result.status == "PASS":
            report.passed += 1
        elif result.status == "FAIL":
            report.failed += 1
        else:
            report.warnings += 1
        
        if verbose:
            status_icon = "✓" if result.status == "PASS" else ("✗" if result.status == "FAIL" else "⚠")
            print(f"  {status_icon} {os.path.basename(filepath)}: {result.status}")
            for issue in result.issues:
                if issue.level in ["ERROR", "WARN"]:
                    print(f"      [{issue.level}] {issue.message}")
    
    # Validar CSV de modos
    if verbose and files["modes_csv"]:
        print(f"\n--- Modes CSV ({len(files['modes_csv'])} archivos) ---")
    for filepath in files["modes_csv"]:
        result = validate_modes_csv(filepath)
        report.files.append(result)
        report.total_files += 1
        if result.status == "PASS":
            report.passed += 1
        elif result.status == "FAIL":
            report.failed += 1
        else:
            report.warnings += 1
        
        if verbose:
            status_icon = "✓" if result.status == "PASS" else ("✗" if result.status == "FAIL" else "⚠")
            print(f"  {status_icon} {os.path.basename(filepath)}: {result.status}")
            for issue in result.issues:
                if issue.level in ["ERROR", "WARN"]:
                    print(f"      [{issue.level}] {issue.message}")
    
    # Validar dictionary JSON
    if verbose and files["dictionary_json"]:
        print(f"\n--- Dictionary JSON ({len(files['dictionary_json'])} archivos) ---")
    for filepath in files["dictionary_json"]:
        result = validate_dictionary_json(filepath)
        report.files.append(result)
        report.total_files += 1
        if result.status == "PASS":
            report.passed += 1
        elif result.status == "FAIL":
            report.failed += 1
        else:
            report.warnings += 1
        
        if verbose:
            status_icon = "✓" if result.status == "PASS" else ("✗" if result.status == "FAIL" else "⚠")
            print(f"  {status_icon} {os.path.basename(filepath)}: {result.status}")
            for issue in result.issues:
                if issue.level in ["ERROR", "WARN"]:
                    print(f"      [{issue.level}] {issue.message}")
    
    # Resumen
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESUMEN")
        print(f"{'='*60}")
        print(f"Total archivos: {report.total_files}")
        print(f"  ✓ PASS: {report.passed}")
        print(f"  ⚠ WARN: {report.warnings}")
        print(f"  ✗ FAIL: {report.failed}")
        
        if report.total_files == 0:
            print(f"\n⚠ No se encontraron archivos para validar en {runs_dir}")
            print(f"  Asegúrate de que exista la estructura:")
            print(f"    - runs/sandbox_geometries/*.h5")
            print(f"    - runs/emergent_geometry/geometry_emergent/*.h5")
            print(f"    - runs/bulk_eigenmodes/*.csv")
        print()
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validador de Contratos IO v1 para CUERDAS-Maldacena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python 00_validate_io_contracts.py --runs-dir runs/
  python 00_validate_io_contracts.py --runs-dir runs/ --output report.json
  python 00_validate_io_contracts.py --runs-dir runs/ --strict
        """
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/",
        help="Directorio raíz de runs (default: runs/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Archivo de salida para el reporte JSON (opcional)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit code 1 si hay cualquier FAIL"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprimir output a consola"
    )
    
    args = parser.parse_args()
    
    report = run_validation(args.runs_dir, verbose=not args.quiet)
    
    # Guardar reporte si se especifica
    if args.output:
        # Convertir dataclasses a dict
        report_dict = asdict(report)
        # Convertir issues a dict
        for file_result in report_dict["files"]:
            file_result["issues"] = [asdict(i) if hasattr(i, '__dataclass_fields__') else i 
                                     for i in file_result["issues"]]
        
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        if not args.quiet:
            print(f"Reporte guardado en: {args.output}")
    
    # Exit code
    if args.strict and report.failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
