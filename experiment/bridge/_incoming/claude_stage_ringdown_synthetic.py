#!/usr/bin/env python3
"""
stage_ringdown_synthetic.py — Generador de features QNM sintéticos para F4-1

Genera un dataset de features de ringdown basado en fórmulas de Kerr QNM
(Berti et al.), parametrizado por masa y spin.

Uso:
    python experiment/bridge/stage_ringdown_synthetic.py \
        --run f4_bridge_pilot \
        --n-points 50 \
        --mass-range 30 100 \
        --spin-range 0.1 0.9 \
        --seed 42

Output:
    runs/<run>/ringdown_synthetic/
        ├── manifest.json
        ├── stage_summary.json
        └── outputs/
            └── features.json

NOTA: Este generador NO introduce teoría holográfica. Solo produce features
      físicos de QNM de Kerr, que luego se comparan agnósticamente con el atlas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# =============================================================================
# Constantes físicas y coeficientes de Berti
# =============================================================================

# Constantes SI
G = 6.67430e-11      # m^3 kg^-1 s^-2
C = 299792458.0      # m/s
MSUN = 1.98892e30    # kg

# Coeficientes de Berti para modo 220 (fundamental)
# f_220 = w_bar / (2π T_g), donde T_g = GM/c³
# w_bar = f1 + f2*(1-spin)^f3
BERTI_F_220 = (1.5251, -1.1568, 0.1292)
BERTI_Q_220 = (0.7000, 1.4187, -0.4990)

# Coeficientes para modo 221 (primer overtone) - aproximados
BERTI_F_221 = (1.2500, -1.0500, 0.1200)
BERTI_Q_221 = (0.4000, 0.8000, -0.4000)


@dataclass(frozen=True)
class Config:
    """Configuración del generador."""
    run: str
    n_points: int = 50
    mass_min: float = 30.0    # Msun
    mass_max: float = 100.0   # Msun
    spin_min: float = 0.1
    spin_max: float = 0.9
    include_overtone: bool = True
    noise_rel: float = 0.0    # Ruido relativo (0 = determinista)
    seed: int = 42


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Genera features QNM sintéticos")
    p.add_argument("--run", required=True, type=str)
    p.add_argument("--n-points", type=int, default=50, dest="n_points")
    p.add_argument("--mass-range", nargs=2, type=float, default=[30.0, 100.0],
                   dest="mass_range", metavar=("MIN", "MAX"))
    p.add_argument("--spin-range", nargs=2, type=float, default=[0.1, 0.9],
                   dest="spin_range", metavar=("MIN", "MAX"))
    p.add_argument("--no-overtone", action="store_true", dest="no_overtone")
    p.add_argument("--noise-rel", type=float, default=0.0, dest="noise_rel")
    p.add_argument("--seed", type=int, default=42)
    
    args = p.parse_args()
    return Config(
        run=args.run,
        n_points=args.n_points,
        mass_min=args.mass_range[0],
        mass_max=args.mass_range[1],
        spin_min=args.spin_range[0],
        spin_max=args.spin_range[1],
        include_overtone=not args.no_overtone,
        noise_rel=args.noise_rel,
        seed=args.seed,
    )


# =============================================================================
# Física de QNM
# =============================================================================

def kerr_qnm(mass_msun: float, spin: float, mode: str = "220") -> tuple[float, float]:
    """Calcula frecuencia y Q para modo QNM de Kerr.
    
    Args:
        mass_msun: masa del agujero negro en masas solares
        spin: parámetro de spin adimensional (0 < spin < 1)
        mode: "220" o "221"
    
    Returns:
        f_hz: frecuencia en Hz
        Q: factor de calidad
    """
    spin = float(np.clip(spin, 0.01, 0.99))
    
    if mode == "220":
        coeffs_f, coeffs_q = BERTI_F_220, BERTI_Q_220
    elif mode == "221":
        coeffs_f, coeffs_q = BERTI_F_221, BERTI_Q_221
    else:
        raise ValueError(f"Modo desconocido: {mode}")
    
    # w_bar adimensional
    w_bar = coeffs_f[0] + coeffs_f[1] * (1 - spin) ** coeffs_f[2]
    Q = coeffs_q[0] + coeffs_q[1] * (1 - spin) ** coeffs_q[2]
    
    # Tiempo gravitacional
    T_g = (G * mass_msun * MSUN) / (C ** 3)
    
    # Frecuencia en Hz
    f_hz = w_bar / (2 * np.pi * T_g)
    
    return float(f_hz), float(max(Q, 0.1))


def compute_qnm_features(mass_msun: float, spin: float, 
                          include_overtone: bool = True) -> dict:
    """Calcula features de QNM para un evento.
    
    Features:
        - f_220: frecuencia del modo fundamental [Hz]
        - Q_220: factor de calidad del modo fundamental
        - f_ratio: f_221 / f_220 (si include_overtone)
        - Q_ratio: Q_221 / Q_220 (si include_overtone)
    
    Returns:
        dict con features y parámetros
    """
    f_220, Q_220 = kerr_qnm(mass_msun, spin, "220")
    
    result = {
        "mass_msun": float(mass_msun),
        "spin": float(spin),
        "f_220_hz": f_220,
        "Q_220": Q_220,
    }
    
    if include_overtone:
        f_221, Q_221 = kerr_qnm(mass_msun, spin, "221")
        result["f_221_hz"] = f_221
        result["Q_221"] = Q_221
        result["f_ratio"] = f_221 / f_220
        result["Q_ratio"] = Q_221 / Q_220
    
    return result


def build_feature_vector(event: dict, include_overtone: bool) -> list[float]:
    """Construye vector de features normalizable.
    
    Vector (con overtone): [log(f_220), Q_220, f_ratio, Q_ratio]
    Vector (sin overtone): [log(f_220), Q_220]
    
    Usamos log(f_220) porque la frecuencia varía en órdenes de magnitud.
    """
    if include_overtone:
        return [
            np.log10(event["f_220_hz"]),
            event["Q_220"],
            event["f_ratio"],
            event["Q_ratio"],
        ]
    else:
        return [
            np.log10(event["f_220_hz"]),
            event["Q_220"],
        ]


# =============================================================================
# IO
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    
    # Crear directorios
    stage_dir = Path("runs") / cfg.run / "ringdown_synthetic"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar sweep de (masa, spin)
    masses = np.linspace(cfg.mass_min, cfg.mass_max, int(np.sqrt(cfg.n_points)))
    spins = np.linspace(cfg.spin_min, cfg.spin_max, int(np.sqrt(cfg.n_points)))
    
    # Grid completo
    events = []
    feature_vectors = []
    
    for i, mass in enumerate(masses):
        for j, spin in enumerate(spins):
            # Calcular features
            event = compute_qnm_features(mass, spin, cfg.include_overtone)
            event["id"] = f"synth_{i:02d}_{j:02d}"
            
            # Añadir ruido si procede
            if cfg.noise_rel > 0:
                for key in ["f_220_hz", "Q_220"]:
                    event[key] *= (1 + np.random.normal(0, cfg.noise_rel))
                if cfg.include_overtone:
                    for key in ["f_221_hz", "Q_221"]:
                        event[key] *= (1 + np.random.normal(0, cfg.noise_rel))
                    event["f_ratio"] = event["f_221_hz"] / event["f_220_hz"]
                    event["Q_ratio"] = event["Q_221"] / event["Q_220"]
            
            events.append(event)
            feature_vectors.append(build_feature_vector(event, cfg.include_overtone))
    
    # Truncar a n_points exactos si sobran
    events = events[:cfg.n_points]
    feature_vectors = feature_vectors[:cfg.n_points]
    
    # Construir output JSON
    if cfg.include_overtone:
        feature_definition = {
            "dim": 4,
            "components": ["log10(f_220_hz)", "Q_220", "f_ratio", "Q_ratio"],
            "description": "Features de QNM Kerr con overtone"
        }
    else:
        feature_definition = {
            "dim": 2,
            "components": ["log10(f_220_hz)", "Q_220"],
            "description": "Features de QNM Kerr sin overtone"
        }
    
    features_data = {
        "version": "1.0.0",
        "generator": "stage_ringdown_synthetic.py",
        "n_events": len(events),
        "feature_definition": feature_definition,
        "parameter_ranges": {
            "mass_msun": [cfg.mass_min, cfg.mass_max],
            "spin": [cfg.spin_min, cfg.spin_max],
        },
        "events": events,
        "feature_matrix": feature_vectors,  # Lista de listas para fácil carga
    }
    
    # Escribir features.json
    features_path = outputs_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump(features_data, f, indent=2)
    
    # Stage summary
    summary = {
        "stage": "ringdown_synthetic",
        "version": "1.0.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "outputs": {
            "features": "outputs/features.json",
        },
        "statistics": {
            "n_events": len(events),
            "f_220_range_hz": [
                min(e["f_220_hz"] for e in events),
                max(e["f_220_hz"] for e in events),
            ],
            "Q_220_range": [
                min(e["Q_220"] for e in events),
                max(e["Q_220"] for e in events),
            ],
        },
        "hashes": {
            "outputs/features.json": sha256_file(features_path),
        },
        "notes": [
            "Features QNM sintéticos basados en fórmulas de Berti et al.",
            "NO contienen información holográfica.",
            "Uso: comparación agnóstica con atlas holográfico.",
        ],
    }
    
    summary_path = stage_dir / "stage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Manifest
    manifest = {
        "stage": "ringdown_synthetic",
        "run": cfg.run,
        "created": datetime.now(timezone.utc).isoformat(),
        "files": {
            "features": "outputs/features.json",
            "summary": "stage_summary.json",
        },
    }
    
    manifest_path = stage_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Console output
    print(f"OK: Features QNM sintéticos generados")
    print(f"  run: {cfg.run}")
    print(f"  n_events: {len(events)}")
    print(f"  mass: [{cfg.mass_min}, {cfg.mass_max}] Msun")
    print(f"  spin: [{cfg.spin_min}, {cfg.spin_max}]")
    print(f"  overtone: {cfg.include_overtone}")
    print(f"  feature_dim: {feature_definition['dim']}")
    print(f"  output: {features_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
