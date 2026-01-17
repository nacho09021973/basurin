#!/usr/bin/env python3
"""
Lee una geometría H5 y verifica que el escalar de Ricci es correcto.

Uso:
    python lee_y_verifica.py --run mi_experimento

Para AdS puro debe imprimir:
    R = -12.00 (esperado: -12.00 para d=3, L=1)
    VERIFICADO: Es AdS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_geometry(h5_path: Path) -> dict:
    """Carga geometría desde H5."""
    try:
        import h5py
    except ImportError:
        print("ERROR: pip install h5py", file=sys.stderr)
        sys.exit(1)
    
    with h5py.File(h5_path, "r") as h5:
        data = {
            "z": h5["z_grid"][:],
            "A": h5["A_of_z"][:],
            "f": h5["f_of_z"][:],
            "d": int(h5.attrs["d"]),
            "L": float(h5.attrs["L"]),
            "family": h5.attrs.get("family", "unknown"),
        }
    return data


def compute_ricci_scalar(z: np.ndarray, A: np.ndarray, f: np.ndarray, d: int) -> np.ndarray:
    """
    Calcula el escalar de Ricci R(z) para la métrica conformal:
    
        ds² = e^{2A(z)} [-f(z)dt² + dx² + dz²/f(z)]
    
    Fórmula (geometría diferencial, no física):
    
        R = e^{-2A} [-2d·A'' - d(d-1)·(A')² - d·(f'/f)·A']
    
    donde d = dimensión de frontera.
    
    Para AdS puro (A = -log(z/L), f = 1):
        R = -d(d+1)/L²  (constante)
    """
    dz = z[1] - z[0]
    
    # Derivadas numéricas (diferencias centradas)
    A_prime = np.gradient(A, dz)
    A_double_prime = np.gradient(A_prime, dz)
    
    f_prime = np.gradient(f, dz)
    f_safe = np.where(np.abs(f) > 1e-10, f, 1e-10)  # evitar división por cero
    
    # Escalar de Ricci
    R = np.exp(-2 * A) * (
        -2 * d * A_double_prime
        - d * (d - 1) * A_prime**2
        - d * (f_prime / f_safe) * A_prime
    )
    
    return R


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica geometría AdS")
    parser.add_argument("--run", type=str, required=True, help="Nombre del run")
    parser.add_argument("--file", type=str, default="ads_puro.h5", help="Archivo H5")
    args = parser.parse_args()
    
    h5_path = Path("runs") / args.run / "geometry" / args.file
    
    if not h5_path.exists():
        print(f"ERROR: No existe {h5_path}", file=sys.stderr)
        return 1
    
    # Cargar
    geo = load_geometry(h5_path)
    z, A, f = geo["z"], geo["A"], geo["f"]
    d, L = geo["d"], geo["L"]
    
    print(f"Archivo: {h5_path}")
    print(f"  d = {d}, L = {L}")
    print(f"  N = {len(z)}, z ∈ [{z[0]:.4f}, {z[-1]:.4f}]")
    print()
    
    # Calcular R
    R = compute_ricci_scalar(z, A, f, d)
    
    # Estadísticas (ignorar bordes por errores numéricos)
    margin = len(z) // 10
    R_interior = R[margin:-margin]
    R_mean = np.mean(R_interior)
    R_std = np.std(R_interior)
    
    # Valor esperado para AdS
    R_expected = -d * (d + 1) / (L * L)
    
    print(f"Escalar de Ricci R(z):")
    print(f"  Media:    {R_mean:.4f}")
    print(f"  Std:      {R_std:.4f}")
    print(f"  Esperado: {R_expected:.4f} (para AdS con d={d}, L={L})")
    print()
    
    # Verificación
    tol_mean = 0.01 * abs(R_expected)  # 1% tolerancia
    tol_std = 0.01 * abs(R_expected)   # 1% variación
    
    is_constant = R_std < tol_std
    is_correct_value = abs(R_mean - R_expected) < tol_mean
    
    if is_constant and is_correct_value:
        print(f"VERIFICADO: R = {R_mean:.2f} es constante y coincide con AdS")
        return 0
    elif is_constant:
        print(f"PARCIAL: R es constante ({R_mean:.2f}) pero no coincide con AdS ({R_expected:.2f})")
        return 1
    else:
        print(f"FALLO: R no es constante (std = {R_std:.4f})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
