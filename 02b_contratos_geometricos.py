"""BASURIN — Contratos geométricos/espectrales post-hoc (Fase 2)

================================================================================
GAUGE / ANSATZ MÉTRICO (CONFIRMADO EN CÓDIGO)
================================================================================

En esta fase, BASURIN representa la geometría mediante arrays numéricos
``[z_grid, A(z), f(z)]`` asumiendo el ansatz:

    ds² = e^{2A(z)} [ -f(z) dt² + d\vec{x}² + dz² / f(z) ]

Esto equivale a la forma más general ``ds² = e^{2A}(-f dt² + dx²) + e^{2B} dz²/f``
con la elección de gauge **B(z)=A(z)**.

Convenciones:
  - ``d`` = dimensión del borde (CFT); el bulk es ``d+1``.
  - ``z`` crece hacia el IR (UV en ``z ≃ z_min``).
  - Cuando se requiere la coordenada radial propia (domain-wall):

        dr/dz = e^{A(z)} / sqrt(f(z))

================================================================================
CLASIFICACIÓN DE CONTRATOS (INTERPRETACIÓN)
================================================================================

Puramente geométricos (no asumen ecuaciones de Einstein):
  - A_uv_ads: asintótica UV ``A ~ -log(z/L) + const`` y ``f ~ 1``.
  - E_horizon: detección de horizonte y raíz simple de ``f``.

Einstein (2 derivadas) + NEC (materia satisface NEC):
  - B_nec: ``A_rr(r) <= 0`` evaluado en coordenada radial propia ``r``.
  - C_c_theorem: c-función estándar ``c(r) ~ 1/|A_r|^{d-1}`` monótona.

Espectral/IR (requiere espectro SL externo):
  - F_regge: ajuste asintótico ``M_n² ~ σ n + b`` (modo alto).

================================================================================
NOTAS DE DISEÑO
================================================================================

1) Los contratos B y C **no invalidan** una geometría en abstracto: invalidan su
   realizabilidad dentro de la clase dinámica declarada (Einstein+NEC).
2) Se adjunta explícitamente ``gauge_assumed`` y ``dynamic_class`` en los
   retornos para auditoría reproducible.

Este módulo no asume IO: opera sobre arrays y devuelve dicts serializables.

Requisitos:
  - numpy
  - scipy es opcional (si está disponible, se usa spline/Savitzky-Golay)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import json
import math
import numpy as np


# -----------------------------
# Metadata (auditoría)
# -----------------------------

GAUGE_ASSUMED = "domain_wall_conformal_z"
GAUGE_DEFINITION = "ds^2 = e^{2A(z)}[-f(z)dt^2 + d\vec{x}^2 + dz^2/f(z)] (B(z)=A(z))"

DYNAMIC_CLASS_GEOMETRY_ONLY = "none"
DYNAMIC_CLASS_EINSTEIN_NEC = "Einstein_2deriv_NEC"


# -----------------------------
# Helpers numéricos
# -----------------------------

def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).ravel()[0])


def is_uniform_grid(z: np.ndarray, rtol: float = 1e-3) -> bool:
    """Comprueba si z es casi-uniforme (necesario para Savitzky-Golay clásico)."""
    z = np.asarray(z)
    dz = np.diff(z)
    if len(dz) < 2:
        return True
    return np.max(np.abs(dz - np.median(dz))) <= rtol * np.median(np.abs(dz))


def odd_window(n: int) -> int:
    """Fuerza a que el tamano de ventana sea impar y >= 5."""
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
    """Derivada suave dy/dz (o d2y/dz2) con fallback robusto.

    - method="auto": Savitzky-Golay si z es uniforme; si no, spline.
    - method="savgol": requiere z ~ uniforme.
    - method="spline": UnivariateSpline.

    Nota: Savitzky-Golay de scipy asume paso uniforme; aquí se usa delta = median(dz).
    """
    z = np.asarray(z)
    y = np.asarray(y)

    if method not in {"auto", "savgol", "spline"}:
        raise ValueError(f"method invalido: {method}")

    use_savgol = (method in {"auto", "savgol"}) and is_uniform_grid(z)

    if use_savgol:
        try:
            from scipy.signal import savgol_filter
        except Exception as e:
            if method == "savgol":
                raise
            use_savgol = False

    if use_savgol:
        w = odd_window(smooth_window)
        w = min(w, len(z) - (len(z) + 1) % 2)  # sigue siendo impar y <= len(z)
        if w < 5:
            w = 5 if len(z) >= 5 else len(z) | 1
        p = min(int(poly_order), w - 1)
        dz = float(np.median(np.diff(z))) if len(z) > 1 else 1.0
        return savgol_filter(y, window_length=w, polyorder=p, deriv=deriv, delta=dz, mode="interp")

    # Spline fallback (admite z no uniforme)
    try:
        from scipy.interpolate import UnivariateSpline
    except Exception as e:
        # último recurso: sin suavizado
        if deriv == 0:
            return y
        if deriv == 1:
            return np.gradient(y, z)
        if deriv == 2:
            return np.gradient(np.gradient(y, z), z)
        raise ValueError("deriv debe ser 0, 1 o 2")

    # Heurística de suavizado:
    # - spline_s=None -> deja que scipy escoja s=0 (interpolante)
    # - si se quiere suavizar: s ~ variance * N * (factor)
    if spline_s is None:
        s = 0.0
    else:
        s = float(spline_s)

    spl = UnivariateSpline(z, y, s=s)
    if deriv == 0:
        return spl(z)
    return spl.derivative(n=deriv)(z)


def interior_mask(z: np.ndarray, margin_frac: float = 0.1) -> np.ndarray:
    """Mascara booleana que excluye margenes UV/IR para evitar artefactos de borde."""
    n = len(z)
    m = int(max(0, math.floor(n * float(margin_frac))))
    mask = np.ones(n, dtype=bool)
    if m > 0:
        mask[:m] = False
        mask[-m:] = False
    return mask


# -----------------------------
# Contratos
# -----------------------------

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
    """Contrato A: asintotica UV AdS en el gauge conforme.

    Checks robustos (evita depender de expresiones largas de R):
    1) Ajuste local A(z) ~ -alpha log(z/L) + beta en UV: alpha ~ 1.
    2) Residuo |A + log(z/L) - beta| pequeño.
    3) f(z) ~ 1 en UV.

    Nota: El check de curvatura (R -> -d(d+1)/L^2) se puede anadir si existe
    un compute_ricci_scalar fiable en el proyecto; aqui no lo imponemos.
    """
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
    # Ajuste lineal: A = -alpha * x + beta
    # polyfit devuelve [m, b] para m*x + b; queremos m ~ -alpha.
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
    """Contrato B: NEC efectiva (solo Einstein 2-derivadas + materia NEC).

    Formulacion invariante (via coordenada radial propia r):
        A_rr <= 0.

    En el gauge conforme:
        A_rr(z) = e^{-2A} * N(z)
        N(z) := f (A'' - (A')^2) + (f'/2) A'

    Como e^{-2A} > 0, el signo de A_rr es el de N(z).
    NEC => N(z) <= 0.

    Implementacion:
    - Derivadas suaves (Savgol si z uniforme; spline si no).
    - Se evalua solo en el interior y donde f > f_floor.

    Retorna estadisticas y sensibilidad al suavizado.
    """
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    # Suavizado (opcional) de señales antes de pasar a coordenada radial propia r
    A0 = smooth_derivative(z, A, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)
    f0 = smooth_derivative(z, f, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)

    # Coordenada radial propia (domain-wall): dr/dz = e^A / sqrt(f)
    fpos = np.maximum(f0, float(f_floor))
    integrand = np.exp(A0) / np.sqrt(fpos)
    dz = np.diff(z)
    # Trapecio acumulado: r[0]=0, r[i]=sum_{j<i} 0.5*(g[j]+g[j+1])*(z[j+1]-z[j])
    r = np.empty_like(z, dtype=float)
    r[0] = 0.0
    r[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)

    # Derivadas respecto a r (no uniforme): A_rr = d^2A/dr^2
    Ar = np.gradient(A0, r)
    Arr = np.gradient(Ar, r)

    # NEC (Einstein+NEC): A_rr <= 0 (en r).
    # Usamos Arr directamente para evitar cancelaciones numéricas en A''-(A')^2.

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

    # Sensibilidad: repetir conteo con distintas ventanas (si procede)
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
    """Contrato C: monotonía de c-función (solo en la clase Einstein+NEC).

    En domain-wall (r), una eleccion estandar:
        c(r) ~ 1 / (A_r)^{d-1}

    Con z conforme:
        A_r(z) = dA/dr = sqrt(f) e^{-A} A'(z)

    Se evalua la monotonía de c respecto a la coordenada radial propia r.
    Por defecto: z crece hacia IR y r(z) es creciente; se espera dc/dr <= 0.

    Nota: si en alguna familia tu convención de dirección es la opuesta,
    basta con invertir el criterio de monotonicidad.
    """
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    # Suavizado (opcional) coherente con NEC: trabajamos con A0, f0
    A0 = smooth_derivative(z, A, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)
    f0 = smooth_derivative(z, f, deriv=0, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)

    # Coordenada radial propia: dr/dz = e^A / sqrt(f)
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

    # A_r = dA/dr
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


def check_bf_bound_uv(m2L2: np.ndarray, d: int) -> Dict[str, Any]:
    """Contrato D: BF bound UV (AdS_{d+1})."""
    m2L2 = np.asarray(m2L2)
    bf = - (int(d) ** 2) / 4.0
    viol = m2L2 < bf
    n_viol = int(np.sum(viol))
    status = "PASS" if n_viol == 0 else "FAIL"
    return {
        "status": status,
        "bf_limit": float(bf),
        "m2L2_min": float(np.min(m2L2)),
        "m2L2_max": float(np.max(m2L2)),
        "margin_to_bf_min": float(np.min(m2L2 - bf)),
        "n_violations": n_viol,
        "contract_class": "spectral_uv",
        "gauge_assumed": GAUGE_ASSUMED,
        "gauge_definition": GAUGE_DEFINITION,
        "dynamic_class": DYNAMIC_CLASS_GEOMETRY_ONLY,
        "requires": ["AdS_UV"],
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
    """Contrato E: regularidad de horizonte si existe f(z_h)=0.

    Checks implementados (robustos y auditables):
    1) Detectar cruce de signo de f.
    2) Raiz simple: |f'(z_h)| > tol_root.
    3) Temperatura finita:
         T = |f'(z_h)| / (4 pi)
       En este gauge (B=A), el factor conformal e^{2A} cancela en la periodicidad.

    Nota: no calcula Kretschmann completo; si se desea, se recomienda
    una etapa simbólica aparte (coste alto) o un proxy basado en acotacion
    de A, f y derivadas en una vecindad del horizonte.
    """
    z = np.asarray(z)
    A = np.asarray(A)
    f = np.asarray(f)

    # Detectar cambio de signo alrededor de cero
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
    # Interpolacion lineal para z_h
    z0, z1 = float(z[i]), float(z[i + 1])
    f0, f1v = float(f[i]), float(f[i + 1])

    if abs(f1v - f0) < 1e-30:
        z_h = z0
    else:
        z_h = z0 - f0 * (z1 - z0) / (f1v - f0)

    # Derivada f'(z)
    fp = smooth_derivative(z, f, deriv=1, method=method, smooth_window=smooth_window, poly_order=poly_order, spline_s=spline_s)
    # Interpolar f' y A en z_h
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
    """Contrato F: comportamiento Regge/soft-wall en IR a partir de M_n^2.

    Se ajusta M_n^2 = sigma * n + b en modos altos.
    - Si M2 es 2D (por ejemplo barrido en delta): se usa la mediana sobre el eje 0.
    """
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


# -----------------------------
# Reporting
# -----------------------------

def build_stage_summary(
    run_id: str,
    gauge: str,
    dynamic_class_assumed: str,
    contracts: Dict[str, Any],
    version: str = "0.1.0",
) -> Dict[str, Any]:
    return {
        "stage": "02_geometric_contracts",
        "version": version,
        "run_id": run_id,
        "gauge": gauge,
        "dynamic_class_assumed": dynamic_class_assumed,
        "contracts": contracts,
        "overall": {
            "geometric_only": ["A_uv_ads", "E_horizon"],
            "require_einstein_nec": ["B_nec", "C_c_theorem"],
            "spectral_ir": ["F_regge"],
        },
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -----------------------------
# CLI minimo (opcional)
# -----------------------------

def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="BASURIN: contratos geometricos post-hoc (sin IO del pipeline)")
    ap.add_argument("--npz", required=True, help="Ruta a un .npz con arrays z,A,f (y opcional M2)")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--run-id", type=str, default="manual")
    ap.add_argument("--out", type=str, default="stage_summary.json")

    args = ap.parse_args()

    data = np.load(args.npz)
    z = data["z"]
    A = data["A"]
    f = data["f"]

    contracts: Dict[str, Any] = {}
    contracts["A_uv_ads"] = check_uv_ads(z, A, f, d=args.d, L=args.L)
    contracts["B_nec"] = check_nec_einstein(z, A, f)
    contracts["C_c_theorem"] = check_c_theorem(z, A, f, d=args.d)
    contracts["E_horizon"] = check_horizon_regularity(z, A, f)

    if "M2" in data:
        contracts["F_regge"] = check_regge_trajectory(data["M2"])

    summary = build_stage_summary(
        run_id=args.run_id,
        gauge="domain_wall_conformal_z",
        dynamic_class_assumed="Einstein_2deriv_NEC",
        contracts=contracts,
    )

    write_json(args.out, summary)


if __name__ == "__main__":
    _cli()
