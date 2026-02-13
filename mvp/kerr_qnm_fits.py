"""Kerr QNM fitting formulas from Berti, Cardoso & Starinets (2009).

Reference: Class. Quant. Grav. 26 (2009) 163001, Table VIII.
           arXiv:0905.2975

Provides f(M, χ) and Q(χ) for modes (2,2,0), (2,2,1), (3,3,0).
Also provides parametric deviations for alternative geometries
(EdGB, dCS, Kerr-Newman) from Gemini research compilation.

Zero external dependencies — uses only Python stdlib + math.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G_SI = 6.67430e-11           # m^3 kg^-1 s^-2
C_SI = 2.99792458e8          # m/s
MSUN_KG = 1.98892e30         # kg
MSUN_S = G_SI * MSUN_KG / C_SI ** 3  # M_sun in seconds ≈ 4.9255e-6 s


# ---------------------------------------------------------------------------
# Berti fits: F_lmn(χ) = f1 + f2*(1-χ)^f3  (dimensionless Mω_real)
#             Q_lmn(χ) = q1 + q2*(1-χ)^q3
# Table VIII, Berti et al. (2009)
# ---------------------------------------------------------------------------
class _FitCoeffs(NamedTuple):
    f1: float
    f2: float
    f3: float
    q1: float
    q2: float
    q3: float


BERTI_FITS: dict[tuple[int, int, int], _FitCoeffs] = {
    (2, 2, 0): _FitCoeffs(f1=1.5251, f2=-1.1568, f3=0.1292,
                           q1=0.7000, q2=1.4187, q3=-0.4990),
    (2, 2, 1): _FitCoeffs(f1=1.3673, f2=-1.0260, f3=0.1628,
                           q1=0.1000, q2=0.5436, q3=-0.4731),
    (3, 3, 0): _FitCoeffs(f1=1.8956, f2=-1.3043, f3=0.1818,
                           q1=0.9000, q2=2.3430, q3=-0.4810),
}

# Maximum spin to avoid divergence in Q fits
CHI_MAX = 0.998


# ---------------------------------------------------------------------------
# Core Kerr QNM functions
# ---------------------------------------------------------------------------

class QNMResult(NamedTuple):
    """Result of a QNM computation."""
    f_hz: float    # Frequency in Hz
    tau_s: float   # Damping time in seconds
    Q: float       # Quality factor (= π f τ)


def kerr_omega_dimless(chi: float, mode: tuple[int, int, int] = (2, 2, 0)) -> float:
    """Dimensionless frequency Re(Mω) for Kerr mode.

    Parameters
    ----------
    chi : float
        Dimensionless spin, 0 <= chi < 1.
    mode : (l, m, n)
        Angular mode numbers.

    Returns
    -------
    float
        F = Re(Mω), dimensionless.
    """
    if mode not in BERTI_FITS:
        raise ValueError(f"No fit for mode {mode}. Available: {list(BERTI_FITS)}")
    c = BERTI_FITS[mode]
    chi = min(chi, CHI_MAX)
    return c.f1 + c.f2 * (1.0 - chi) ** c.f3


def kerr_Q(chi: float, mode: tuple[int, int, int] = (2, 2, 0)) -> float:
    """Quality factor Q for Kerr mode.

    Parameters
    ----------
    chi : float
        Dimensionless spin, 0 <= chi < 1.
    mode : (l, m, n)
        Angular mode numbers.

    Returns
    -------
    float
        Q = π f τ (dimensionless).
    """
    if mode not in BERTI_FITS:
        raise ValueError(f"No fit for mode {mode}. Available: {list(BERTI_FITS)}")
    c = BERTI_FITS[mode]
    chi = min(chi, CHI_MAX)
    return c.q1 + c.q2 * (1.0 - chi) ** c.q3


def kerr_qnm(M_solar: float, chi: float,
              mode: tuple[int, int, int] = (2, 2, 0)) -> QNMResult:
    """Compute physical QNM (f, τ, Q) for a Kerr black hole.

    Parameters
    ----------
    M_solar : float
        BH mass in solar masses.
    chi : float
        Dimensionless spin, 0 <= chi < 1.
    mode : (l, m, n)
        Angular mode numbers.

    Returns
    -------
    QNMResult(f_hz, tau_s, Q)
    """
    F = kerr_omega_dimless(chi, mode)
    Q_val = kerr_Q(chi, mode)

    M_s = M_solar * MSUN_S
    f_hz = F / (2.0 * math.pi * M_s)
    tau_s = Q_val / (math.pi * f_hz)

    return QNMResult(f_hz=f_hz, tau_s=tau_s, Q=Q_val)


# ---------------------------------------------------------------------------
# Alternative geometries: parametric deviations from Kerr
#
# Each returns (delta_f, delta_tau) as fractional shifts:
#   f = f_Kerr * (1 + delta_f)
#   tau = tau_Kerr * (1 + delta_tau)
#
# Sources compiled from Gemini research (see references in docstrings).
# ---------------------------------------------------------------------------

def deviation_edgb(chi: float, zeta: float) -> tuple[float, float]:
    """Einstein-dilaton-Gauss-Bonnet fractional shifts.

    Ref: Blazquez-Salcedo et al. (2016) PRD 94, 104024;
         Maselli et al. (2020) PRL 124, 171101.

    Parameters
    ----------
    chi : float
        Spin (enters weakly; these are chi~0 leading-order coefficients).
    zeta : float
        Dimensionless EdGB coupling parameter.

    Returns
    -------
    (delta_f, delta_tau) fractional shifts.
    """
    delta_f = -0.027 * zeta ** 2
    delta_tau = 0.083 * zeta ** 2
    return delta_f, delta_tau


def deviation_dcs(chi: float, zeta: float) -> tuple[float, float]:
    """Dynamical Chern-Simons fractional shifts.

    Ref: Wagle et al. (2022) PRD 105, 124003.

    Note: dCS has no effect at chi=0 (parity-violating).

    Parameters
    ----------
    chi : float
        Dimensionless spin.
    zeta : float
        Dimensionless dCS coupling.

    Returns
    -------
    (delta_f, delta_tau) fractional shifts.
    """
    delta_f = 0.038 * zeta * chi ** 2
    # delta_Q = ... but we have delta_tau from Gemini: -0.668 * zeta * chi^2
    delta_tau = -0.668 * zeta * chi ** 2
    return delta_f, delta_tau


def deviation_kerr_newman(chi: float, q_charge: float) -> tuple[float, float]:
    """Kerr-Newman (charged BH) fractional shifts.

    Ref: Dias et al. (2015) PRD 92, 084023.
    Approximate for small q = Q_charge/M.

    Parameters
    ----------
    chi : float
        Dimensionless spin.
    q_charge : float
        Dimensionless charge Q/M.

    Returns
    -------
    (delta_f, delta_tau) fractional shifts.
    """
    # sigma_f ≈ 0.2, sigma_Q ≈ -0.1 at chi~0.7
    # delta_tau ≈ -delta_Q (to first order, since Q = π f τ)
    delta_f = 0.2 * q_charge ** 2
    delta_tau = 0.1 * q_charge ** 2  # +0.1 because delta_Q ~ -0.1
    return delta_f, delta_tau


def apply_deviation(
    base: QNMResult,
    delta_f: float,
    delta_tau: float,
) -> QNMResult:
    """Apply fractional shifts to a Kerr QNM result.

    Returns a new QNMResult with shifted f, tau, and recomputed Q.
    """
    f_new = base.f_hz * (1.0 + delta_f)
    tau_new = base.tau_s * (1.0 + delta_tau)
    Q_new = math.pi * f_new * tau_new
    return QNMResult(f_hz=f_new, tau_s=tau_new, Q=Q_new)


# ---------------------------------------------------------------------------
# Atlas entry builder
# ---------------------------------------------------------------------------

def make_atlas_entry(
    geometry_id: str,
    theory: str,
    qnm: QNMResult,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an atlas entry dict compatible with s4_geometry_filter."""
    phi = [math.log(qnm.f_hz), math.log(qnm.Q)] if qnm.f_hz > 0 and qnm.Q > 0 else None
    entry: dict[str, Any] = {
        "geometry_id": geometry_id,
        "theory": theory,
        "f_hz": qnm.f_hz,
        "tau_s": qnm.tau_s,
        "Q": qnm.Q,
        "phi_atlas": phi,
    }
    if metadata:
        entry["metadata"] = metadata
    return entry
