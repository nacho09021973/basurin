#!/usr/bin/env python3
"""
stage_bridge_discovery.py — Motor principal de Fase 4 (F4-1)

Descubre y evalúa puentes estructurales entre atlas holográfico y features externos.

Uso:
    python experiment/bridge/stage_bridge_discovery.py \
        --run f4_bridge_pilot \
        --atlas runs/f4_bridge_pilot/dictionary/outputs/atlas.json \
        --external runs/f4_bridge_pilot/ringdown_synthetic/outputs/features.json \
        --method cca \
        --k-neighbors 5 \
        --n-bootstrap 100 \
        --seed 42

Output:
    runs/<run>/bridge/
        ├── manifest.json
        ├── stage_summary.json
        └── outputs/
            ├── bridge_results.json
            ├── degeneracy_analysis.json
            ├── stability_analysis.json
            ├── projections.npz
            └── controls/
                ├── positive_control.json
                └── negative_control.json

Contratos evaluados:
    C7a: Estructura (kNN preservation)
    C7b: No-degeneración
    C7c: Estabilidad (bootstrap)
    C7d: No-falso-positivo (control negativo)
    C7e: Control positivo (split atlas)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Import local
try:
    from bridge_metrics import (
        evaluate_bridge_full, 
        normalize_features,
        generate_negative_control,
        knn_preservation,
        cca_bridge,
    )
except ImportError:
    # Si se ejecuta desde raíz del proyecto
    sys.path.insert(0, str(Path(__file__).parent))
    from bridge_metrics import (
        evaluate_bridge_full,
        normalize_features,
        generate_negative_control,
        knn_preservation,
        cca_bridge,
    )


__version__ = "1.0.0"


@dataclass(frozen=True)
class Config:
    run: str
    atlas_path: str
    external_path: str
    method: str = "cca"
    k_neighbors: int = 5
    n_bootstrap: int = 100
    n_components: Optional[int] = None
    seed: int = 42
    # Umbrales (pueden ser CLI args en futuras versiones)
    threshold_knn: float = 0.3
    threshold_degen: float = 0.5
    threshold_cv: float = 0.3


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="F4-1: Bridge Discovery")
    p.add_argument("--run", required=True, type=str)
    p.add_argument("--atlas", required=True, type=str, dest="atlas_path",
                   help="Path a atlas.json del diccionario")
    p.add_argument("--external", required=True, type=str, dest="external_path",
                   help="Path a features.json del dataset externo")
    p.add_argument("--method", type=str, default="cca", 
                   choices=["cca", "procrustes"],
                   help="Método de puente (default: cca)")
    p.add_argument("--k-neighbors", type=int, default=5, dest="k_neighbors")
    p.add_argument("--n-bootstrap", type=int, default=100, dest="n_bootstrap")
    p.add_argument("--n-components", type=int, default=None, dest="n_components")
    p.add_argument("--seed", type=int, default=42)
    
    args = p.parse_args()
    return Config(
        run=args.run,
        atlas_path=args.atlas_path,
        external_path=args.external_path,
        method=args.method,
        k_neighbors=args.k_neighbors,
        n_bootstrap=args.n_bootstrap,
        n_components=args.n_components,
        seed=args.seed,
    )


def load_atlas(path: str | Path) -> tuple[np.ndarray, list[dict]]:
    """Carga atlas y extrae features (ratios).
    
    Returns:
        X_A: array (n_theories, k) con ratios
        theories: lista de dicts con metadata
    """
    with open(path, "r") as f:
        atlas = json.load(f)
    
    theories = atlas["theories"]
    # Extraer ratios como matriz
    X_A = np.array([t["ratios"] for t in theories], dtype=np.float64)
    
    return X_A, theories


def load_external_features(path: str | Path) -> tuple[np.ndarray, list[dict]]:
    """Carga features externos.
    
    Returns:
        X_B: array (n_events, dim) con features
        events: lista de dicts con metadata
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    # Buscar feature_matrix primero, si no, construir desde events
    if "feature_matrix" in data:
        X_B = np.array(data["feature_matrix"], dtype=np.float64)
    else:
        # Asumir que events tiene un campo 'features'
        events = data.get("events", [])
        X_B = np.array([e["features"] for e in events], dtype=np.float64)
    
    events = data.get("events", [])
    return X_B, events


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    
    # Verificar inputs
    atlas_path = Path(cfg.atlas_path)
    external_path = Path(cfg.external_path)
    
    if not atlas_path.exists():
        print(f"ERROR: No existe atlas en {atlas_path}", file=sys.stderr)
        return 1
    if not external_path.exists():
        print(f"ERROR: No existe external en {external_path}", file=sys.stderr)
        return 1
    
    print(f"F4-1: Bridge Discovery v{__version__}")
    print(f"  run: {cfg.run}")
    print(f"  atlas: {atlas_path}")
    print(f"  external: {external_path}")
    print(f"  method: {cfg.method}")
    print()
    
    # Cargar datos
    X_A, theories = load_atlas(atlas_path)
    X_B, events = load_external_features(external_path)
    
    print(f"Datos cargados:")
    print(f"  Atlas: {X_A.shape[0]} puntos, {X_A.shape[1]} features")
    print(f"  External: {X_B.shape[0]} puntos, {X_B.shape[1]} features")
    print()
    
    # Verificar compatibilidad de tamaños
    n_A, n_B = X_A.shape[0], X_B.shape[0]
    n_common = min(n_A, n_B)
    
    if n_A != n_B:
        print(f"WARN: Tamaños diferentes ({n_A} vs {n_B}). Usando primeros {n_common} puntos de cada uno.")
        X_A = X_A[:n_common]
        X_B = X_B[:n_common]
    
    if n_common < 10:
        print(f"ERROR: Muy pocos puntos ({n_common}). Necesito al menos 10.", file=sys.stderr)
        return 1
    
    # Crear directorios de output
    stage_dir = Path("runs") / cfg.run / "bridge"
    outputs_dir = stage_dir / "outputs"
    controls_dir = outputs_dir / "controls"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)
    
    # Ejecutar evaluación completa
    print("Evaluando puente...")
    result = evaluate_bridge_full(
        X_A, X_B,
        k_neighbors=cfg.k_neighbors,
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.seed,
    )
    
    # Extraer proyecciones para guardar
    A_proj = result["projections"]["A_proj"]
    B_proj = result["projections"]["B_proj"]
    
    # Guardar proyecciones
    proj_path = outputs_dir / "projections.npz"
    np.savez(proj_path, A_proj=A_proj, B_proj=B_proj)
    
    # Preparar bridge_results.json (sin arrays grandes)
    bridge_results = {
        "version": __version__,
        "method": cfg.method,
        "status": result["status"],
        "failure_mode": result["failure_mode"],
        "contracts": result["contracts"],
        "diagnostics": result["diagnostics"],
        "config": {
            "k_neighbors": cfg.k_neighbors,
            "n_bootstrap": cfg.n_bootstrap,
            "seed": cfg.seed,
        },
        "data_shapes": {
            "X_A": list(X_A.shape),
            "X_B": list(X_B.shape),
            "A_proj": list(A_proj.shape),
            "B_proj": list(B_proj.shape),
        },
    }
    
    results_path = outputs_dir / "bridge_results.json"
    with open(results_path, "w") as f:
        json.dump(bridge_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    # Guardar análisis de degeneración detallado
    degen_analysis = {
        "detail_A": result["contracts"]["C7b_degeneracy"]["detail_A"],
        "detail_B": result["contracts"]["C7b_degeneracy"]["detail_B"],
    }
    degen_path = outputs_dir / "degeneracy_analysis.json"
    with open(degen_path, "w") as f:
        json.dump(degen_analysis, f, indent=2)
    
    # Guardar análisis de estabilidad
    stability_analysis = {
        "bootstrap_mean": result["contracts"]["C7c_stability"]["bootstrap_mean"],
        "bootstrap_std": result["contracts"]["C7c_stability"]["bootstrap_std"],
        "bootstrap_cv": result["contracts"]["C7c_stability"]["bootstrap_cv"],
    }
    stability_path = outputs_dir / "stability_analysis.json"
    with open(stability_path, "w") as f:
        json.dump(stability_analysis, f, indent=2)
    
    # Guardar controles
    pos_control = {
        "type": "positive_split",
        "knn_preservation": result["contracts"]["C7e_positive_control"]["control_pos_knn"],
        "status": result["contracts"]["C7e_positive_control"]["status"],
    }
    pos_path = controls_dir / "positive_control.json"
    with open(pos_path, "w") as f:
        json.dump(pos_control, f, indent=2)
    
    neg_control = {
        "type": "permutation",
        "knn_preservation": result["contracts"]["C7d_no_false_positive"]["control_neg_knn"],
        "status": result["contracts"]["C7d_no_false_positive"]["status"],
    }
    neg_path = controls_dir / "negative_control.json"
    with open(neg_path, "w") as f:
        json.dump(neg_control, f, indent=2)
    
    # Stage summary
    summary = {
        "stage": "bridge_discovery",
        "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "inputs": {
            "atlas": cfg.atlas_path,
            "external": cfg.external_path,
        },
        "results": {
            "status": result["status"],
            "failure_mode": result["failure_mode"],
            "contracts_summary": {
                k: v["status"] for k, v in result["contracts"].items()
            },
        },
        "hashes": {
            "outputs/bridge_results.json": sha256_file(results_path),
            "outputs/projections.npz": sha256_file(proj_path),
        },
    }
    
    summary_path = stage_dir / "stage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Manifest
    manifest = {
        "stage": "bridge_discovery",
        "run": cfg.run,
        "created": datetime.now(timezone.utc).isoformat(),
        "files": {
            "bridge_results": "outputs/bridge_results.json",
            "degeneracy_analysis": "outputs/degeneracy_analysis.json",
            "stability_analysis": "outputs/stability_analysis.json",
            "projections": "outputs/projections.npz",
            "positive_control": "outputs/controls/positive_control.json",
            "negative_control": "outputs/controls/negative_control.json",
            "summary": "stage_summary.json",
        },
        "inputs": {
            "atlas": cfg.atlas_path,
            "external": cfg.external_path,
        },
    }
    
    manifest_path = stage_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Console report
    print()
    print("=" * 60)
    print("RESULTADOS F4-1: Bridge Discovery")
    print("=" * 60)
    print()
    print(f"STATUS GLOBAL: {result['status']}")
    if result["failure_mode"]:
        print(f"FAILURE MODE:  {result['failure_mode']}")
    print()
    print("Contratos:")
    for name, contract in result["contracts"].items():
        status = contract["status"]
        marker = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        print(f"  {marker} {name}: {status}")
        # Mostrar métricas clave
        if "knn_preservation" in contract:
            print(f"      kNN preservation: {contract['knn_preservation']:.4f}")
        if "degeneracy_index" in contract:
            print(f"      Degeneracy index: {contract['degeneracy_index']:.4f}")
        if "bootstrap_cv" in contract:
            print(f"      Bootstrap CV: {contract['bootstrap_cv']:.4f}")
    
    print()
    print("Diagnósticos:")
    diag = result["diagnostics"]
    if "canonical_correlations" in diag:
        print(f"  Correlaciones canónicas: {diag['canonical_correlations']}")
    if "explained_variance_A" in diag:
        print(f"  Varianza explicada (A): {diag['explained_variance_A']:.4f}")
        print(f"  Varianza explicada (B): {diag['explained_variance_B']:.4f}")
    
    print()
    print("Outputs:")
    print(f"  {results_path}")
    print(f"  {proj_path}")
    print(f"  {manifest_path}")
    print()
    
    # Return code
    if result["status"] == "PASS":
        print("[OK] Bridge discovery: PASS")
        return 0
    elif result["status"] in ["FAIL_STRUCTURE", "FAIL_DEGENERACY"]:
        print(f"[FAIL] Bridge discovery: {result['status']} — resultado informativo")
        return 0  # Es un FAIL informativo, no un error
    else:
        print(f"[FAIL] Bridge discovery: {result['status']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
