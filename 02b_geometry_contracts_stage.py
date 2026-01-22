#!/usr/bin/env python3
"""BASURIN — Stage: Contratos geométricos/espectrales post-hoc (Fase 2)

Este stage evalúa validadores *post-hoc* sobre la geometría reconstruida y,
si existe, sobre el espectro SL (Bloque B).

IO (determinista, contrato BASURIN):
  runs/<run>/geometry_contracts/
    - manifest.json
    - stage_summary.json
    - outputs/
        - contracts.json

Inputs esperados:
  - Geometría (canónico): runs/<run>/geometry/outputs/<geometry-file>  (H5)
  - Geometría (legacy, solo lectura): runs/<run>/geometry/<geometry-file>
    datasets: z_grid, A_of_z, f_of_z; attrs: d, L
  - (Opcional) Espectro: runs/<run>/spectrum/outputs/spectrum.h5
    datasets: M2 (o M2_D), attrs: d, L

Gauge (confirmado):
  ds^2 = e^{2A(z)}[-f(z)dt^2 + d\vec{x}^2 + dz^2/f(z)]

Notas teóricas importantes:
  - NEC y c-theorem se evalúan de forma invariante en coordenada radial propia r,
    implementado sin reparametrizar manualmente toda la malla.

Uso:
  python 02b_geometry_contracts_stage.py --run <run_id> [--geometry-file ads_puro.h5]

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from basurin_io import (
    ensure_stage_dirs,
    get_run_dir,
    resolve_geometry_path,
    resolve_spectrum_path,
    sha256_file,
    write_manifest,
    write_stage_summary,
)

_HAS_SCIPY_SIGNAL = find_spec("scipy.signal") is not None
_HAS_SCIPY_INTERP = find_spec("scipy.interpolate") is not None

if _HAS_SCIPY_SIGNAL:
    from scipy.signal import savgol_filter

if _HAS_SCIPY_INTERP:
    from scipy.interpolate import UnivariateSpline


# -----------------------------
# Utilidades IO BASURIN
# -----------------------------

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -----------------------------
# Lectura de inputs
# -----------------------------

def load_geometry(h5_path: Path) -> Dict[str, Any]:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        z = h5["z_grid"][:]
        A = h5["A_of_z"][:]
        f = h5["f_of_z"][:]
        d = int(h5.attrs["d"])
        L = float(h5.attrs["L"])
        family = h5.attrs.get("family", "unknown")

    return {
        "z": z,
        "A": A,
        "f": f,
        "d": d,
        "L": L,
        "N": int(len(z)),
        "z_min": float(z[0]),
        "z_max": float(z[-1]),
        "family": family,
    }


def load_spectrum_M2(h5_path: Path) -> Optional[np.ndarray]:
    """Carga M2 para contrato Regge. Devuelve None si no hay archivo."""
    if not h5_path.exists():
        return None

    import h5py

    with h5py.File(h5_path, "r") as h5:
        if "M2" in h5:
            return h5["M2"][:]
        if "M2_D" in h5:
            return h5["M2_D"][:]

    return None


# -----------------------------
# Geometría diferencial mínima
# -----------------------------

GAUGE_ASSUMED = "domain_wall_conformal_z"
GAUGE_DEFINITION = "ds^2=e^{2A}[-f dt^2+d\\vec{x}^2+dz^2/f] (B=A)"
DYNAMIC_CLASS_GEOMETRY_ONLY = "none"
DYNAMIC_CLASS_EINSTEIN_NEC = "Einstein_2deriv_NEC"


def is_uniform_grid(z: np.ndarray, rtol: float = 1e-3) -> bool:
    z = np.asarray(z)
    dz = np.diff(z)
    if len(dz) < 2:
        return True
    return np.max(np.abs(dz - np.median(dz))) <= rtol * np.median(np.abs(dz))


def odd_window(n: int) -> int:
    n = int(n)
    if n < 5:
        n = 5
    if n % 2 == 0:
        n += 1
    return n


def smooth_derivative(
    z: np.ndarray,
    y: np.ndarray,
    deriv: int = 1,
    method: str = "auto",
    smooth_window: int = 11,
    poly_order: int = 3,
    spline_s: Optional[float] = None,
) -> np.ndarray:
    z = np.asarray(z)
    y = np.asarray(y)

    if method not in {"auto", "savgol", "spline"}:
        raise ValueError(f"method invalido: {method}")

    use_savgol = (method in {"auto", "savgol"}) and is_uniform_grid(z) and _HAS_SCIPY_SIGNAL

    if use_savgol:
        w = odd_window(smooth_window)
        w = min(w, len(z) - (len(z) + 1) % 2)
        if w < 5:
            w = 5 if len(z) >= 5 else len(z) | 1
        p = min(int(poly_order), w - 1)
        dz = float(np.median(np.diff(z))) if len(z) > 1 else 1.0
        return savgol_filter(y, window_length=w, polyorder=p, deriv=deriv, delta=dz, mode="interp")

    if method == "savgol" and not _HAS_SCIPY_SIGNAL:
        raise RuntimeError("scipy.signal no disponible para Savitzky-Golay")

    if _HAS_SCIPY_INTERP:
        s = 0.0 if spline_s is None else float(spline_s)
        spl = UnivariateSpline(z, y, s=s)
        if deriv == 0:
            return spl(z)
        return spl.derivative(n=deriv)(z)

    if deriv == 0:
        return y
    if deriv == 1:
        return np.gradient(y, z)
    if deriv == 2:
        return np.gradient(np.gradient(y, z), z)
    raise ValueError("deriv debe ser 0, 1 o 2")


def interior_mask(z: np.ndarray, margin_frac: float = 0.1) -> np.ndarray:
    n = len(z)
    m = int(max(0, math.floor(n * float(margin_frac))))
    mask = np.ones(n, dtype=bool)
    if m > 0:
        mask[:m] = False
        mask[-m:] = False
    return mask


def check_uv_ads(
    z: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    d: int,
    L: float,
    n_uv_points: int = 20,
    tol_alpha: float = 0.02,
    tol_resid: float = 0.05,
    tol_f: float = 0.02,
) -> Dict[str, Any]:
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    n = min(int(n_uv_points), len(z))
    if n < 5:
        return {
            "status": "SKIP",
            "reason": "insuficientes puntos UV",
            "contract_class": "geometric_pure",
            "gauge_assumed": GAUGE_ASSUMED,
            "gauge_definition": GAUGE_DEFINITION,
            "dynamic_class": DYNAMIC_CLASS_GEOMETRY_ONLY,
            "requires": [],
        }

    z_uv = z[:n]
    A_uv = A[:n]
    f_uv = f[:n]

    x = np.log(z_uv / float(L))
    m, b = np.polyfit(x, A_uv, deg=1)
    alpha = -float(m)

    A_fit = m * x + b
    resid = A_uv - A_fit
    f_err = np.abs(f_uv - 1.0)

    status = "PASS"
    if abs(alpha - 1.0) > tol_alpha:
        status = "FAIL"
    if float(np.max(np.abs(resid))) > tol_resid:
        status = "FAIL"
    if float(np.median(f_err)) > tol_f:
        status = "FAIL"

    return {
        "status": status,
        "n_uv_points": int(n),
        "alpha_log": float(alpha),
        "beta": float(b),
        "resid_abs_max": float(np.max(np.abs(resid))),
        "resid_abs_median": float(np.median(np.abs(resid))),
        "f_abs_median_err": float(np.median(f_err)),
        "f_abs_max_err": float(np.max(f_err)),
        "notes": "Check robusto UV: A ~ -log(z/L) + const, f~1",
        "contract_class": "geometric_pure",
        "gauge_assumed": GAUGE_ASSUMED,
        "gauge_definition": GAUGE_DEFINITION,
        "dynamic_class": DYNAMIC_CLASS_GEOMETRY_ONLY,
        "requires": [],
    }


def check_nec_einstein(
    z: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    smooth_window: int = 11,
    poly_order: int = 3,
    method: str = "auto",
    spline_s: Optional[float] = None,
    margin_frac: float = 0.1,
    f_floor: float = 1e-6,
    tol: float = 0.0,
    sensitivity_windows: Tuple[int, ...] = (7, 11, 15, 21),
) -> Dict[str, Any]:
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    A0 = smooth_derivative(z, A, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)
    f0 = smooth_derivative(z, f, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)

    fpos = np.maximum(f0, float(f_floor))
    integrand = np.exp(A0) / np.sqrt(fpos)
    dz = np.diff(z)
    r = np.empty_like(z, dtype=float)
    r[0] = 0.0
    r[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)

    Ar = np.gradient(A0, r)
    Arr = np.gradient(Ar, r)

    mask = interior_mask(z, margin_frac=margin_frac) & (f > float(f_floor))
    Arr_in = Arr[mask]

    if Arr_in.size == 0:
        return {
            "status": "SKIP",
            "reason": "mascara vacia (margenes o f_floor)",
            "contract_class": "einstein_nec",
            "gauge_assumed": GAUGE_ASSUMED,
            "gauge_definition": GAUGE_DEFINITION,
            "dynamic_class": DYNAMIC_CLASS_EINSTEIN_NEC,
            "requires": ["Einstein_2deriv", "NEC"],
            "caveat": "Valida realizabilidad Einstein(2-deriv)+NEC; no invalida geometria fuera de esa clase",
        }

    viol = Arr_in > float(tol)
    n_viol = int(np.sum(viol))
    max_viol = float(np.max(Arr_in[viol])) if n_viol else 0.0

    sensitivity: Dict[str, Any] = {}
    for w in sensitivity_windows:
        w = int(w)
        if w >= 5 and w < len(z):
            try:
                A0t = smooth_derivative(z, A, deriv=0, method=method, smooth_window=w, poly_order=poly_order, spline_s=spline_s)
                f0t = smooth_derivative(z, f, deriv=0, method=method, smooth_window=w, poly_order=poly_order, spline_s=spline_s)
                fpos_t = np.maximum(f0t, float(f_floor))
                integ_t = np.exp(A0t) / np.sqrt(fpos_t)
                r_t = np.empty_like(z, dtype=float)
                r_t[0] = 0.0
                r_t[1:] = np.cumsum(0.5 * (integ_t[:-1] + integ_t[1:]) * np.diff(z))
                Ar_t = np.gradient(A0t, r_t)
                Arr_t = np.gradient(Ar_t, r_t)
                Arr_in_t = Arr_t[mask]
                sensitivity[f"window_{w}"] = int(np.sum(Arr_in_t > float(tol)))
            except Exception:
                sensitivity[f"window_{w}"] = "error"

    status = "PASS" if n_viol == 0 else "FAIL"

    return {
        "status": status,
        "n_violations": n_viol,
        "fraction_violating": float(n_viol / Arr_in.size),
        "max_violation": max_viol,
        "Arr_median": float(np.median(Arr_in)),
        "Arr_p95": float(np.percentile(Arr_in, 95)),
        "smooth_window": int(smooth_window),
        "poly_order": int(poly_order),
        "method": method,
        "sensitivity_to_smoothing": sensitivity,
        "contract_class": "einstein_nec",
        "gauge_assumed": GAUGE_ASSUMED,
        "gauge_definition": GAUGE_DEFINITION,
        "dynamic_class": DYNAMIC_CLASS_EINSTEIN_NEC,
        "requires": ["Einstein_2deriv", "NEC"],
        "caveat": "Valida realizabilidad Einstein(2-deriv)+NEC; no invalida geometria fuera de esa clase",
        "definition": "NEC (Einstein+NEC): A_rr(r) <= 0 con dr/dz = e^A/sqrt(f); Arr se estima con derivadas numéricas respecto a r",
    }


def check_c_theorem(
    z: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    d: int,
    smooth_window: int = 11,
    poly_order: int = 3,
    method: str = "auto",
    spline_s: Optional[float] = None,
    margin_frac: float = 0.1,
    f_floor: float = 1e-6,
    eps: float = 1e-12,
    tol: float = 0.0,
) -> Dict[str, Any]:
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    A0 = smooth_derivative(z, A, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)
    f0 = smooth_derivative(z, f, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)

    fpos = np.maximum(f0, float(f_floor))
    integrand = np.exp(A0) / np.sqrt(fpos)
    dz = np.diff(z)
    r = np.empty_like(z, dtype=float)
    r[0] = 0.0
    r[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)

    mask = interior_mask(z, margin_frac=margin_frac) & (f0 > float(f_floor))
    if int(np.sum(mask)) < 5:
        return {
            "status": "SKIP",
            "reason": "mascara vacia",
            "contract_class": "einstein_nec",
            "gauge_assumed": GAUGE_ASSUMED,
            "gauge_definition": GAUGE_DEFINITION,
            "dynamic_class": DYNAMIC_CLASS_EINSTEIN_NEC,
            "requires": ["Einstein_2deriv", "NEC"],
            "caveat": "Valida monotonía c-función en clase Einstein(2-deriv)+NEC",
        }

    Ar = np.gradient(A0, r)
    Ar_in = Ar[mask]
    r_in = r[mask]

    denom = np.maximum(np.abs(Ar_in), float(eps))
    c = 1.0 / (denom ** (int(d) - 1))
    dc_dr = np.gradient(c, r_in)

    viol = dc_dr > float(tol)
    n_viol = int(np.sum(viol))
    status = "PASS" if n_viol == 0 else "FAIL"

    return {
        "status": status,
        "c_UV": float(c[0]),
        "c_IR": float(c[-1]),
        "ratio_c_IR_UV": float(c[-1] / c[0]) if c[0] != 0 else float("nan"),
        "n_violations": n_viol,
        "fraction_violating": float(n_viol / len(dc_dr)),
        "dc_dr_p95": float(np.percentile(dc_dr, 95)),
        "smooth_window": int(smooth_window),
        "poly_order": int(poly_order),
        "method": method,
        "contract_class": "einstein_nec",
        "gauge_assumed": GAUGE_ASSUMED,
        "gauge_definition": GAUGE_DEFINITION,
        "dynamic_class": DYNAMIC_CLASS_EINSTEIN_NEC,
        "requires": ["Einstein_2deriv", "NEC"],
        "caveat": "Valida monotonía c-función en clase Einstein(2-deriv)+NEC; no invalida geometría fuera de esa clase",
        "definition": "c(r)~1/|A_r|^{d-1} con A_r=dA/dr, r definida por dr/dz=e^A/sqrt(f); exige dc/dr<=0 hacia el IR",
    }


def check_horizon_regularity(
    z: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    smooth_window: int = 11,
    poly_order: int = 3,
    method: str = "auto",
    spline_s: Optional[float] = None,
    tol_root: float = 1e-8,
) -> Dict[str, Any]:
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    s = np.sign(f)
    s[s == 0] = 1.0
    idx = np.where(np.diff(s) != 0)[0]

    if len(idx) == 0:
        return {
            "status": "NO_HORIZON",
            "has_horizon": False,
            "note": "f no cruza 0",
            "contract_class": "geometric_pure",
            "gauge_assumed": GAUGE_ASSUMED,
            "gauge_definition": GAUGE_DEFINITION,
            "dynamic_class": DYNAMIC_CLASS_GEOMETRY_ONLY,
            "requires": [],
        }

    i = int(idx[0])
    z0, z1 = float(z[i]), float(z[i + 1])
    f0, f1v = float(f[i]), float(f[i + 1])

    if abs(f1v - f0) < 1e-30:
        z_h = z0
    else:
        z_h = z0 - f0 * (z1 - z0) / (f1v - f0)

    fp = smooth_derivative(z, f, deriv=1, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)
    fp_h = float(np.interp(z_h, z, fp))
    A_h = float(np.interp(z_h, z, A))

    is_simple = abs(fp_h) > float(tol_root)
    T = abs(fp_h) / (4.0 * math.pi)

    status = "PASS" if (is_simple and T > 0.0) else "FAIL"

    return {
        "status": status,
        "has_horizon": True,
        "z_horizon": float(z_h),
        "f_prime_at_horizon": float(fp_h),
        "is_simple_root": bool(is_simple),
        "T_hawking": float(T),
        "A_at_horizon": float(A_h),
        "contract_class": "geometric_pure",
        "gauge_assumed": GAUGE_ASSUMED,
        "gauge_definition": GAUGE_DEFINITION,
        "dynamic_class": DYNAMIC_CLASS_GEOMETRY_ONLY,
        "requires": [],
        "definition": "horizon: f(z_h)=0, simple root |f'(z_h)|>tol, T=|f'(z_h)|/(4pi)",
    }


def check_regge_trajectory(
    M2: np.ndarray,
    n_skip_low: int = 3,
    r2_threshold: float = 0.95,
) -> Dict[str, Any]:
    M2 = np.asarray(M2)

    if M2.ndim == 2:
        y = np.median(M2, axis=0)
    else:
        y = M2

    n_modes = len(y)
    n_skip_low = int(n_skip_low)

    if n_modes < max(8, n_skip_low + 4):
        return {
            "status": "SKIP",
            "reason": f"insuficientes modos: {n_modes}",
            "contract_class": "spectral_ir",
            "gauge_assumed": GAUGE_ASSUMED,
            "gauge_definition": GAUGE_DEFINITION,
            "dynamic_class": DYNAMIC_CLASS_GEOMETRY_ONLY,
            "requires": ["spectrum_SL"],
        }

    x = np.arange(n_skip_low, n_modes)
    yf = y[n_skip_low:]

    coeffs = np.polyfit(x, yf, deg=1)
    sigma = float(coeffs[0])
    intercept = float(coeffs[1])

    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((yf - y_pred) ** 2))
    ss_tot = float(np.sum((yf - float(np.mean(yf))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    is_regge = (r2 >= float(r2_threshold)) and (sigma > 0.0)

    return {
        "status": "PASS" if is_regge else "INCONCLUSIVE",
        "sigma_regge": sigma,
        "intercept": intercept,
        "r2": float(r2),
        "n_modes_fitted": int(len(x)),
        "n_skip_low": int(n_skip_low),
        "r2_threshold": float(r2_threshold),
        "contract_class": "spectral_ir",
        "gauge_assumed": GAUGE_ASSUMED,
        "gauge_definition": GAUGE_DEFINITION,
        "dynamic_class": DYNAMIC_CLASS_GEOMETRY_ONLY,
        "requires": ["spectrum_SL"],
        "definition": "fit M_n^2 ~ sigma*n + b at high n",
    }

def compute_ricci_scalar(z: np.ndarray, A: np.ndarray, f: np.ndarray, d: int) -> np.ndarray:
    """Escalar de Ricci para el gauge conforme.

    R = e^{-2A}[-2 d A'' - d(d-1)(A')^2 - d (f'/f) A']
    """
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    # Permite z no uniforme
    A1 = np.gradient(A, z)
    A2 = np.gradient(A1, z)
    f1 = np.gradient(f, z)
    f_safe = np.where(np.abs(f) > 1e-12, f, 1e-12)

    R = np.exp(-2.0 * A) * (
        -2.0 * d * A2
        - d * (d - 1) * (A1 ** 2)
        - d * (f1 / f_safe) * A1
    )
    return R


def ricci_uv_stats(
    z: np.ndarray,
    A: np.ndarray,
    f: np.ndarray,
    d: int,
    L: float,
    n_uv_points: int = 20,
    *,
    gauge_assumed: str = "domain_wall_conformal_z",
    gauge_definition: str = "ds^2=e^{2A}[-f dt^2+dx^2+dz^2/f] (B=A)",
    dynamic_class: str = "none",
) -> Dict[str, Any]:
    n = min(int(n_uv_points), len(z))
    if n < 5:
        return {
            "status": "SKIP",
            "reason": "insuficientes puntos UV",
            "contract_class": "geometric_pure",
            "gauge_assumed": gauge_assumed,
            "gauge_definition": gauge_definition,
            "dynamic_class": dynamic_class,
            "requires": [],
        }

    R = compute_ricci_scalar(z[:n], A[:n], f[:n], d)
    R_expected = -d * (d + 1) / (L * L)

    rel = np.abs(R - R_expected) / max(abs(R_expected), 1e-30)

    status = "PASS" if float(np.median(rel)) < 0.01 else "FAIL"
    return {
        "status": status,
        "R_expected": float(R_expected),
        "R_median": float(np.median(R)),
        "R_p10": float(np.percentile(R, 10)),
        "R_p90": float(np.percentile(R, 90)),
        "rel_error_R_median": float(np.median(rel)),
        "rel_error_R_p90": float(np.percentile(rel, 90)),
        "n_uv_points": int(n),
        "notes": "Check UV Ricci ~ -d(d+1)/L^2 (sensibles a discretización; usar como auditor complementario)",
        "contract_class": "geometric_pure",
        "gauge_assumed": gauge_assumed,
        "gauge_definition": gauge_definition,
        "dynamic_class": dynamic_class,
        "requires": [],
    }


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Config:
    run: str
    geometry_file: str = "ads_puro.h5"
    # Contracts params
    n_uv_points: int = 20
    margin_frac: float = 0.10
    smooth_window: int = 11
    poly_order: int = 3
    method: str = "auto"  # auto|savgol|spline (en el módulo)
    spline_s: Optional[float] = None
    f_floor: float = 1e-6
    nec_tol: float = 0.0
    c_tol: float = 0.0
    regge_skip_low: int = 3
    regge_r2_threshold: float = 0.95

    # Declarativos
    gauge: str = "domain_wall_conformal_z"
    dynamic_class_assumed: str = "Einstein_2deriv_NEC"  # para B,C


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="BASURIN: stage de contratos geométricos post-hoc")
    p.add_argument("--run", required=True, type=str, help="Nombre del run")
    p.add_argument("--geometry-file", default="ads_puro.h5", type=str, dest="geometry_file")

    # Contratos
    p.add_argument("--n-uv-points", default=20, type=int, dest="n_uv_points")
    p.add_argument("--margin-frac", default=0.10, type=float, dest="margin_frac")
    p.add_argument("--smooth-window", default=11, type=int, dest="smooth_window")
    p.add_argument("--poly-order", default=3, type=int, dest="poly_order")
    p.add_argument("--method", default="auto", choices=["auto", "savgol", "spline"], dest="method")
    p.add_argument("--spline-s", default=None, type=float, dest="spline_s")
    p.add_argument("--f-floor", default=1e-6, type=float, dest="f_floor")
    p.add_argument("--nec-tol", default=0.0, type=float, dest="nec_tol")
    p.add_argument("--c-tol", default=0.0, type=float, dest="c_tol")
    p.add_argument("--regge-skip-low", default=3, type=int, dest="regge_skip_low")
    p.add_argument("--regge-r2-threshold", default=0.95, type=float, dest="regge_r2_threshold")

    args = p.parse_args()
    return Config(**{k: v for k, v in vars(args).items()})


# -----------------------------
# Stage
# -----------------------------

def run_stage(cfg: Config) -> Dict[str, Path]:
    run_dir = get_run_dir(cfg.run)
    geo_path, geometry_path, input_geometry_absolute, geometry_resolution = resolve_geometry_path(
        cfg.run, cfg.geometry_file
    )

    geo = load_geometry(geo_path)
    z, A, f = geo["z"], geo["A"], geo["f"]
    d, L = geo["d"], geo["L"]

    # Optional spectrum
    try:
        spec_path = resolve_spectrum_path(run_dir)
    except FileNotFoundError as exc:
        spec_path = None
        missing_reason = str(exc)
        M2 = None
    else:
        missing_reason = None
        M2 = load_spectrum_M2(spec_path)

    # Contratos
    contracts: Dict[str, Any] = {}

    contracts["A_uv_ads"] = check_uv_ads(
        z, A, f, d=d, L=L, n_uv_points=cfg.n_uv_points
    )
    # Auditor complementario: Ricci UV
    contracts["A_uv_ricci"] = ricci_uv_stats(
        z,
        A,
        f,
        d=d,
        L=L,
        n_uv_points=cfg.n_uv_points,
        gauge_assumed=cfg.gauge,
        gauge_definition="ds^2=e^{2A}[-f dt^2+d\vec{x}^2+dz^2/f] (B=A)",
        dynamic_class="none",
    )

    contracts["B_nec"] = check_nec_einstein(
        z,
        A,
        f,
        smooth_window=cfg.smooth_window,
        poly_order=cfg.poly_order,
        method=cfg.method,
        spline_s=cfg.spline_s,
        margin_frac=cfg.margin_frac,
        f_floor=cfg.f_floor,
        tol=cfg.nec_tol,
    )

    contracts["C_c_theorem"] = check_c_theorem(
        z,
        A,
        f,
        d=d,
        smooth_window=cfg.smooth_window,
        poly_order=cfg.poly_order,
        method=cfg.method,
        spline_s=cfg.spline_s,
        margin_frac=cfg.margin_frac,
        f_floor=cfg.f_floor,
        tol=cfg.c_tol,
    )

    contracts["E_horizon"] = check_horizon_regularity(
        z,
        A,
        f,
        smooth_window=cfg.smooth_window,
        poly_order=cfg.poly_order,
        method=cfg.method,
        spline_s=cfg.spline_s,
    )

    spectrum_rel = None
    if spec_path is not None:
        spectrum_rel = (
            "../spectrum/outputs/spectrum.h5"
            if "outputs" in spec_path.parts
            else "../spectrum/spectrum.h5"
        )

    if M2 is None:
        reason = missing_reason or f"No existe {spec_path}"
        contracts["F_regge"] = {"status": "SKIP", "reason": reason}
    else:
        regge = check_regge_trajectory(M2, n_skip_low=cfg.regge_skip_low, r2_threshold=cfg.regge_r2_threshold)
        regge["spectrum_source"] = spectrum_rel
        contracts["F_regge"] = regge

    # IO
    stage_dir, outputs_dir = ensure_stage_dirs(cfg.run, "geometry_contracts")

    contracts_path = outputs_dir / "contracts.json"
    write_json(contracts_path, contracts)

    # stage_summary
    summary = {
        "stage": "02b_geometry_contracts",
        "version": "0.1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "geometry": {
            "d": d,
            "L": L,
            "N": geo["N"],
            "z_min": geo["z_min"],
            "z_max": geo["z_max"],
            "family": geo["family"],
            "source": geometry_path,
        },
        "inputs": {
            "geometry_h5": geometry_path,
            "spectrum_h5": spectrum_rel,
            "geometry_resolution": geometry_resolution,
            **(
                {"input_geometry_absolute": input_geometry_absolute}
                if input_geometry_absolute
                else {}
            ),
        },
        "contracts": {k: v.get("status", "UNKNOWN") if isinstance(v, dict) else "UNKNOWN" for k, v in contracts.items()},
        "contract_groups": {
            "geometric_pure": ["A_uv_ads", "A_uv_ricci", "E_horizon"],
            "einstein_nec": ["B_nec", "C_c_theorem"],
            "spectral_ir": ["F_regge"],
        },
        "interpretation_notes": {
            "geometric_pure": "No asume ecuaciones de Einstein; checks de consistencia geométrica y gauge.",
            "einstein_nec": "Asume Einstein (2 derivadas) + materia que satisface NEC; valida realizabilidad dentro de esa clase.",
            "spectral_ir": "Depende de espectro SL (Bloque B) y diagnostica asintótica IR (Regge/soft-wall).",
        },
        "hashes": {
            "outputs/contracts.json": sha256_file(contracts_path),
        },
    }

    summary_path = write_stage_summary(stage_dir, summary)

    manifest_path = write_manifest(
        stage_dir,
        {
            "contracts": contracts_path,
            "summary": summary_path,
        },
        extra={
            "input_geometry": geometry_path,
            "input_spectrum": spectrum_rel,
            "geometry_resolution": geometry_resolution,
            **(
                {"input_geometry_absolute": input_geometry_absolute}
                if input_geometry_absolute
                else {}
            ),
        },
    )

    return {
        "contracts": contracts_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> int:
    cfg = parse_args()
    try:
        out = run_stage(cfg)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("=== Contratos geométricos post-hoc (Fase 2) ===")
    print(f"run: {cfg.run}")
    print(f"outputs: {out['contracts']}")
    print(f"summary:  {out['summary']}")
    print(f"manifest: {out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
