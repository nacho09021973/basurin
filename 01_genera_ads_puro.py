#!/usr/bin/env python3
"""
Genera geometría bulk de AdS puro y la guarda en HDF5.

Uso:
    python genera_ads_puro.py --run mi_experimento

Salida:
    runs/mi_experimento/geometry/ads_puro.h5

El H5 contiene:
    - z_grid: array (N,)
    - A_of_z: array (N,), A(z) = -log(z/L)
    - f_of_z: array (N,), f(z) = 1
    - attrs: d, L, z_min, z_max, N, family
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    run: str
    z_min: float
    z_max: float
    n: int
    d: int
    L: float


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Genera AdS puro en HDF5")
    p.add_argument("--run", type=str, required=True, help="Nombre del run (obligatorio)")
    p.add_argument("--z-min", type=float, default=0.01, dest="z_min")
    p.add_argument("--z-max", type=float, default=1.0, dest="z_max")
    p.add_argument("--n", type=int, default=512, help="Puntos en z")
    p.add_argument("--d", type=int, default=3, help="Dimensión frontera")
    p.add_argument("--L", type=float, default=1.0, help="Radio AdS")
    
    args = p.parse_args()
    return Config(
        run=args.run,
        z_min=args.z_min,
        z_max=args.z_max,
        n=args.n,
        d=args.d,
        L=args.L,
    )


def validate(cfg: Config) -> None:
    if not cfg.run:
        raise ValueError("--run es obligatorio")
    if cfg.z_min <= 0:
        raise ValueError("z_min debe ser > 0")
    if cfg.z_max <= cfg.z_min:
        raise ValueError("z_max debe ser > z_min")
    if cfg.n < 10:
        raise ValueError("n debe ser >= 10")
    if cfg.d < 2:
        raise ValueError("d debe ser >= 2")
    if cfg.L <= 0:
        raise ValueError("L debe ser > 0")


def main() -> int:
    cfg = parse_args()
    validate(cfg)
    
    # Crear directorio
    out_dir = Path("runs") / cfg.run / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ads_puro.h5"
    
    # Generar datos
    z = np.linspace(cfg.z_min, cfg.z_max, cfg.n, dtype=np.float64)
    A = -np.log(z / cfg.L)
    f = np.ones_like(z)
    
    # Verificar antes de escribir
    assert np.all(np.isfinite(z)), "z contiene NaN/Inf"
    assert np.all(np.isfinite(A)), "A contiene NaN/Inf"
    assert np.all(z > 0), "z debe ser positivo"
    
    # Escribir H5
    try:
        import h5py
    except ImportError:
        print("ERROR: pip install h5py", file=sys.stderr)
        return 1
    
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("z_grid", data=z)
        h5.create_dataset("A_of_z", data=A)
        h5.create_dataset("f_of_z", data=f)
        
        h5.attrs["family"] = "ads"
        h5.attrs["d"] = cfg.d
        h5.attrs["L"] = cfg.L
        h5.attrs["z_min"] = cfg.z_min
        h5.attrs["z_max"] = cfg.z_max
        h5.attrs["N"] = cfg.n
        h5.attrs["created"] = datetime.now(timezone.utc).isoformat()
    
    print(f"OK: {out_path}")
    print(f"    d={cfg.d}, L={cfg.L}, N={cfg.n}, z=[{cfg.z_min}, {cfg.z_max}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
