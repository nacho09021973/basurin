#!/usr/bin/env python3
"""Build enriched multi-event feature table for symbolic discovery.

Reads:
  - gwtc_quality_events.csv     : GWTC catalog (56 events)
  - gwtc_events_t0.json         : GPS + posteriors
  - docs/ringdown/event_metadata/*.json : per-event classification

Computes all derived physical quantities (mass ratios, Kerr QNM predictions
via Berti fits, black-hole area/entropy proxies, dimensionless combinations).

Outputs:
  - malda/runs/feature_table/event_features.h5
  - malda/runs/feature_table/event_features.csv
  - malda/runs/feature_table/feature_catalog.json  (column descriptions)

Usage:
    python malda/10_build_event_feature_table.py [--out-dir DIR]

Design philosophy:
    NO physics assumptions injected beyond what's already in the catalog.
    All "derived" quantities are algebraically exact (mass ratios, area formula,
    Berti QNM fits). PySR/KAN will see raw numbers — they discover structure,
    not confirm it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = REPO_ROOT / "gwtc_quality_events.csv"
T0_JSON = REPO_ROOT / "gwtc_events_t0.json"
META_DIR = REPO_ROOT / "docs" / "ringdown" / "event_metadata"

# Physical constants
MSUN_S = 4.925491025543576e-6   # G*M_sun/c^3  [seconds]
CHI_MAX = 0.998                  # numerical limit for Berti fits

# ---------------------------------------------------------------------------
# Berti QNM fits (Berti, Cardoso & Starinets 2009, Table VIII)
# F_lmn(chi) = f1 + f2*(1-chi)^f3     dimensionless Re(M*omega)
# Q_lmn(chi) = q1 + q2*(1-chi)^q3
# ---------------------------------------------------------------------------
BERTI: dict[tuple[int, int, int], tuple[float, float, float, float, float, float]] = {
    (2, 2, 0): (1.5251, -1.1568, 0.1292,  0.7000,  1.4187, -0.4990),
    (2, 2, 1): (1.3673, -1.0260, 0.1628,  0.1000,  0.5436, -0.4731),
    (3, 3, 0): (1.8956, -1.3043, 0.1818,  0.9000,  2.3430, -0.4810),
}


def berti_F(chi: float, mode: tuple[int, int, int]) -> float:
    f1, f2, f3, *_ = BERTI[mode]
    return f1 + f2 * (1.0 - min(chi, CHI_MAX)) ** f3


def berti_Q(chi: float, mode: tuple[int, int, int]) -> float:
    *_, q1, q2, q3 = BERTI[mode]
    return q1 + q2 * (1.0 - min(chi, CHI_MAX)) ** q3


def qnm_hz(Mf_msun: float, af: float, mode: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return (f_hz, tau_s, Q) for a Kerr BH with mass Mf_msun, spin af."""
    F = berti_F(af, mode)
    Q = berti_Q(af, mode)
    M_s = Mf_msun * MSUN_S
    f_hz = F / (2.0 * math.pi * M_s)
    tau_s = Q / (math.pi * f_hz)
    return f_hz, tau_s, Q


# ---------------------------------------------------------------------------
# Final spin estimator
# The GWTC CSV "final_spin" column is actually the lower uncertainty on
# final_mass_source (mislabeled). We derive af from (eta, chi_eff) using
# the Barausse-Rezzolla (2009) / Rezzolla et al. (2008) non-precessing fit.
# Accurate to ~5-10% for aligned-spin BBH mergers.
# ---------------------------------------------------------------------------

def estimate_af(eta: float, chi_eff: float) -> float:
    """Estimate final BH dimensionless spin from symmetric mass ratio + chi_eff.

    Uses Rezzolla et al. (2008) orbital AM piece + leading-order spin correction.
    Returns NaN if inputs are not finite or out of range.
    """
    if not (math.isfinite(eta) and math.isfinite(chi_eff)):
        return float("nan")
    if eta <= 0 or eta > 0.25:
        return float("nan")
    # Orbital AM piece (NR polynomial fit, Rezzolla 2008 Eq. A1 coefficients)
    a_orb = eta * (3.4641016 + eta * (-4.3994 + eta * (6.4296 + eta * (-6.0237))))
    # Leading-order spin correction (Damour 2001 / HBR 2016)
    a_spin = chi_eff * (1.0 - 2.0 * eta)
    af = a_orb + a_spin
    return float(min(max(af, 0.0), CHI_MAX))


# ---------------------------------------------------------------------------
# BH area (geometric units, G=c=1)
#   A = 16 pi M^2 (1 + sqrt(1 - a^2))
# We keep the dimensionless shape factor xi = (1 + sqrt(1-a^2))
# so area in M_sun^2 = 16*pi*M^2*xi
# ---------------------------------------------------------------------------

def bh_xi(af: float) -> float:
    """Dimensionless area shape factor (= 2 for Schwarzschild, 1 for extremal)."""
    a = min(abs(af), CHI_MAX)
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def bh_entropy_proxy(M_msun: float, af: float) -> float:
    """S ∝ M^2 * xi(a).  Proportional to Bekenstein-Hawking entropy (G=c=ħ=kB=1)."""
    return M_msun ** 2 * bh_xi(af)


# ---------------------------------------------------------------------------
# CSV reader (no pandas dependency for the builder itself)
# ---------------------------------------------------------------------------

def read_catalog_csv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        rows.append(dict(zip(header, parts)))
    return rows


def safe_float(value: str | float | None, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        v = float(str(value).strip())
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Per-event feature extraction
# ---------------------------------------------------------------------------

def load_metadata(event_id: str) -> dict[str, Any]:
    path = META_DIR / f"{event_id}_metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def classify_source(meta: dict[str, Any]) -> dict[str, int]:
    sc = str(meta.get("source_class", "")).lower()
    return {
        "is_bbh": int("black_hole" in sc and "neutron" not in sc),
        "is_bns": int("neutron_star" in sc and "black" not in sc),
        "is_nsbh": int("neutron" in sc and "black" in sc),
        "has_multimessenger": int(bool(meta.get("multimessenger", False))),
    }


def build_row(
    cat: dict[str, str],
    t0_entry: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {}

    # --- Catalog scalars ---
    event_id = cat.get("event", "")
    row["event_id"] = event_id
    row["GPS"] = safe_float(cat.get("GPS") or t0_entry.get("GPS"))
    row["snr"] = safe_float(cat.get("snr") or t0_entry.get("snr"))
    row["p_astro"] = safe_float(cat.get("p_astro") or t0_entry.get("p_astro"))
    row["far_yr"] = safe_float(cat.get("far_yr"))

    # Masses (source frame, solar masses)
    m1 = safe_float(cat.get("m1_source") or t0_entry.get("m1"))
    m2 = safe_float(cat.get("m2_source") or t0_entry.get("m2"))
    Mtotal = safe_float(cat.get("M_total_source") or t0_entry.get("M_total"))
    Mchirp = safe_float(cat.get("chirp_mass_source") or t0_entry.get("chirp_mass"))
    chi_eff = safe_float(cat.get("chi_eff") or t0_entry.get("chi_eff"))
    Mf = safe_float(cat.get("final_mass_source") or t0_entry.get("Mf"))
    # NOTE: the CSV column named "final_spin" is actually the lower uncertainty
    # on final_mass_source (mislabeled). We estimate af from (eta, chi_eff) below.
    DL = safe_float(cat.get("luminosity_distance") or t0_entry.get("DL_Mpc"))
    z = safe_float(cat.get("redshift") or t0_entry.get("z"))

    row["m1_src"] = m1
    row["m2_src"] = m2
    row["M_total"] = Mtotal if math.isfinite(Mtotal) else (m1 + m2 if math.isfinite(m1) and math.isfinite(m2) else float("nan"))
    row["Mchirp"] = Mchirp
    row["chi_eff"] = chi_eff
    row["Mf"] = Mf
    row["DL_Mpc"] = DL
    row["z"] = z

    M_total_used = float(row["M_total"])

    # --- Derived: mass ratios ---
    if math.isfinite(m1) and math.isfinite(m2) and m1 > 0 and m2 > 0:
        # Convention: m1 >= m2, so q <= 1
        m_heavy = max(m1, m2)
        m_light = min(m1, m2)
        q = m_light / m_heavy
        eta = m_light * m_heavy / (m_light + m_heavy) ** 2
        delta = (m_heavy - m_light) / (m_heavy + m_light)
        row["q"] = q                         # mass ratio [0..1]
        row["eta"] = eta                     # symmetric mass ratio [0..0.25]
        row["delta"] = delta                 # mass asymmetry [0..1]
        row["log_q"] = math.log(q)          # log(q), always <= 0
    else:
        row["q"] = float("nan")
        row["eta"] = float("nan")
        row["delta"] = float("nan")
        row["log_q"] = float("nan")

    # --- Estimated final spin (from inspiral parameters) ---
    # CSV "final_spin" column is mislabeled (it's lower error on final_mass).
    # We estimate af using Rezzolla et al. (2008) + HBR (2016) fitting formula.
    eta_val = float(row.get("eta", float("nan")))
    chi_eff_val = float(row.get("chi_eff", float("nan")))
    af = estimate_af(eta_val, chi_eff_val)
    row["af"] = af

    # --- Derived: radiated energy ---
    if math.isfinite(Mf) and math.isfinite(M_total_used) and M_total_used > 0:
        E_rad = M_total_used - Mf
        row["E_rad_Msun"] = E_rad
        row["E_rad_frac"] = E_rad / M_total_used   # fraction of total mass radiated
    else:
        row["E_rad_Msun"] = float("nan")
        row["E_rad_frac"] = float("nan")

    # --- Kerr QNM predictions from Berti fits ---
    # Only meaningful if Mf and af are available and valid
    if math.isfinite(Mf) and math.isfinite(af) and Mf > 0 and 0.0 <= af < 1.0:
        for mode, label in [((2,2,0), "220"), ((2,2,1), "221"), ((3,3,0), "330")]:
            f_hz, tau_s, Q = qnm_hz(Mf, af, mode)
            row[f"f_{label}_hz"] = f_hz
            row[f"tau_{label}_s"] = tau_s
            row[f"Q_{label}"] = Q
            # Dimensionless Re(M*omega): spin-dependent factor only
            row[f"F_{label}_dimless"] = berti_F(af, mode)

        # Frequency / quality-factor ratios (mass-independent in Kerr)
        f220 = float(row["f_220_hz"])
        f221 = float(row["f_221_hz"])
        Q220 = float(row["Q_220"])
        Q221 = float(row["Q_221"])
        row["f_ratio_221_220"] = f221 / f220 if f220 > 0 else float("nan")
        row["Q_ratio_221_220"] = Q221 / Q220 if Q220 > 0 else float("nan")

        # Dimensionless frequency: f_220 * Mf [in geometric time units]
        row["Mf_f220_dimless"] = f_hz * Mf * MSUN_S * 2.0 * math.pi  # = 2*pi*M*f = F_220

    else:
        for label in ["220", "221", "330"]:
            for col in [f"f_{label}_hz", f"tau_{label}_s", f"Q_{label}", f"F_{label}_dimless"]:
                row[col] = float("nan")
        for col in ["f_ratio_221_220", "Q_ratio_221_220", "Mf_f220_dimless"]:
            row[col] = float("nan")

    # --- BH area / entropy (G=c=1 units) ---
    # A_BH = 16*pi*G^2*M^2*(1 + sqrt(1-a^2)) / c^4
    # We keep dimensionless shape xi and track everything in solar mass^2 units.
    if math.isfinite(Mf) and math.isfinite(af) and Mf > 0:
        xi_f = bh_xi(af)
        row["xi_f"] = xi_f                              # area shape factor of final BH
        row["S_f"] = bh_entropy_proxy(Mf, af)           # ~ Bekenstein entropy [M_sun^2]

    else:
        row["xi_f"] = float("nan")
        row["S_f"] = float("nan")

    # Schwarzschild (a=0) entropy estimates for the progenitors
    if math.isfinite(m1) and math.isfinite(m2):
        S1 = m1 ** 2   # xi(0) = 2, but we absorb the factor; use just M^2 for ratio purposes
        S2 = m2 ** 2
        row["S1_schw"] = S1
        row["S2_schw"] = S2
        Sf = float(row.get("S_f", float("nan")))
        if math.isfinite(Sf):
            # Hawking area theorem: delta_S = S_f - S_1 - S_2 >= 0
            # Here xi(0)=2 so Schwarzschild entropy ∝ 2*M^2; final ∝ xi_f*Mf^2
            # We compute delta using xi factors explicitly:
            S1_full = 2.0 * m1 ** 2
            S2_full = 2.0 * m2 ** 2
            S_f_full = xi_f * Mf ** 2 if math.isfinite(float(row.get("xi_f", float("nan")))) else float("nan")
            if math.isfinite(S_f_full):
                delta_S = S_f_full - S1_full - S2_full
                row["delta_S"] = delta_S                        # should be >= 0
                row["delta_S_frac"] = delta_S / (S1_full + S2_full)
                row["S_ratio"] = S_f_full / (S1_full + S2_full) # should be >= 1
            else:
                row["delta_S"] = float("nan")
                row["delta_S_frac"] = float("nan")
                row["S_ratio"] = float("nan")
        else:
            row["delta_S"] = float("nan")
            row["delta_S_frac"] = float("nan")
            row["S_ratio"] = float("nan")
    else:
        row["S1_schw"] = float("nan")
        row["S2_schw"] = float("nan")
        row["delta_S"] = float("nan")
        row["delta_S_frac"] = float("nan")
        row["S_ratio"] = float("nan")

    # --- Chirp-mass derived ---
    if math.isfinite(Mchirp) and math.isfinite(M_total_used) and M_total_used > 0:
        row["Mchirp_over_Mtotal"] = Mchirp / M_total_used   # = eta^(3/5)

        # Reconstruct eta from chirp mass if not already computed
        if not math.isfinite(float(row.get("eta", float("nan")))):
            # Mchirp = eta^(3/5) * M_total => eta = (Mchirp/M_total)^(5/3)
            row["eta_from_chirp"] = (Mchirp / M_total_used) ** (5.0 / 3.0)
    else:
        row["Mchirp_over_Mtotal"] = float("nan")

    # --- Distance / cosmology ---
    if math.isfinite(DL) and DL > 0:
        row["log_DL"] = math.log(DL)
    else:
        row["log_DL"] = float("nan")
    if math.isfinite(z) and z > 0:
        row["log1pz"] = math.log(1.0 + z)
    else:
        row["log1pz"] = float("nan")

    # --- Source classification ---
    row.update(classify_source(meta))
    row["glitch_mitigated"] = int(str(cat.get("glitch_mitigated", "False")).strip() == "True")

    # catalog origin
    row["catalog"] = cat.get("catalog", "")

    return row


# ---------------------------------------------------------------------------
# Feature catalog (human-readable column descriptions)
# ---------------------------------------------------------------------------
FEATURE_CATALOG: dict[str, str] = {
    "event_id": "LIGO/Virgo event identifier",
    "GPS": "GPS time of coalescence",
    "snr": "Network signal-to-noise ratio",
    "p_astro": "Probability of astrophysical origin",
    "far_yr": "False alarm rate [yr^-1]",
    "m1_src": "Primary mass (source frame) [M_sun]",
    "m2_src": "Secondary mass (source frame) [M_sun]",
    "M_total": "Total initial mass (source frame) [M_sun]",
    "Mchirp": "Chirp mass [M_sun]",
    "chi_eff": "Effective inspiral spin parameter",
    "Mf": "Final remnant mass [M_sun]",
    "af": "Final remnant dimensionless spin (estimated from eta+chi_eff via Rezzolla+2008/HBR+2016)",
    "DL_Mpc": "Luminosity distance [Mpc]",
    "z": "Redshift",
    "q": "Mass ratio m_light/m_heavy [0..1]",
    "eta": "Symmetric mass ratio m1*m2/(m1+m2)^2 [0..0.25]",
    "delta": "Mass asymmetry (m1-m2)/(m1+m2)",
    "log_q": "log(q), natural log of mass ratio",
    "E_rad_Msun": "Energy radiated in gravitational waves [M_sun c^2]",
    "E_rad_frac": "Fraction of total mass radiated",
    "f_220_hz": "Predicted f_220 QNM frequency [Hz] (Berti+2009 Kerr fit)",
    "tau_220_s": "Predicted tau_220 damping time [s] (Kerr fit)",
    "Q_220": "Quality factor pi*f*tau for (2,2,0) mode",
    "F_220_dimless": "Dimensionless Re(M*omega) for (2,2,0): spin-only factor",
    "f_221_hz": "Predicted f_221 QNM frequency [Hz] (Kerr fit)",
    "tau_221_s": "Predicted tau_221 damping time [s]",
    "Q_221": "Quality factor for (2,2,1) overtone",
    "F_221_dimless": "Dimensionless Re(M*omega) for (2,2,1)",
    "f_330_hz": "Predicted f_330 QNM frequency [Hz]",
    "tau_330_s": "Predicted tau_330 damping time [s]",
    "Q_330": "Quality factor for (3,3,0) mode",
    "F_330_dimless": "Dimensionless Re(M*omega) for (3,3,0)",
    "f_ratio_221_220": "f_221/f_220 frequency ratio (mass-independent in Kerr)",
    "Q_ratio_221_220": "Q_221/Q_220 quality factor ratio",
    "Mf_f220_dimless": "2*pi*Mf[s]*f_220: should equal F_220_dimless",
    "xi_f": "BH area shape factor 1+sqrt(1-af^2) [2=Schwarzschild, 1=extremal]",
    "S_f": "Final BH entropy proxy Mf^2*(1+sqrt(1-af^2)) [M_sun^2, G=c=1]",
    "S1_schw": "Schwarzschild entropy proxy for m1 [M_sun^2]",
    "S2_schw": "Schwarzschild entropy proxy for m2 [M_sun^2]",
    "delta_S": "Entropy increase S_f_full - S1_full - S2_full [M_sun^2] (Hawking: >=0)",
    "delta_S_frac": "Fractional entropy increase delta_S/(S1+S2)",
    "S_ratio": "S_f/(S1+S2): ratio of final to initial entropy (Hawking: >=1)",
    "Mchirp_over_Mtotal": "Mchirp/Mtotal = eta^(3/5)",
    "log_DL": "log(DL/Mpc)",
    "log1pz": "log(1+z)",
    "is_bbh": "Flag: binary black hole merger",
    "is_bns": "Flag: binary neutron star merger",
    "is_nsbh": "Flag: neutron star - black hole merger",
    "has_multimessenger": "Flag: electromagnetic counterpart detected",
    "glitch_mitigated": "Flag: glitch mitigation applied",
    "catalog": "GWTC catalog version",
}


# ---------------------------------------------------------------------------
# HDF5 writer (pure h5py)
# ---------------------------------------------------------------------------

def write_hdf5(rows: list[dict], path: Path) -> None:
    try:
        import h5py
    except ImportError:
        print("[10_build] h5py not available — skipping HDF5 output", file=sys.stderr)
        return

    # Separate numeric and string columns
    all_keys = list(FEATURE_CATALOG.keys())
    float_keys = []
    str_keys = ["event_id", "catalog"]

    for k in all_keys:
        if k not in str_keys:
            float_keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["description"] = "BASURIN enriched GW event feature table"
        f.attrs["n_events"] = len(rows)

        # Float datasets
        for col in float_keys:
            arr = np.array([float(row.get(col, float("nan"))) for row in rows], dtype=np.float64)
            ds = f.create_dataset(col, data=arr)
            ds.attrs["description"] = FEATURE_CATALOG.get(col, "")

        # String datasets (use object dtype to avoid h5py Unicode issues)
        for col in str_keys:
            dt = h5py.string_dtype(encoding="utf-8")
            arr = np.array([str(row.get(col, "")) for row in rows], dtype=object)
            ds = f.create_dataset(col, data=arr, dtype=dt)
            ds.attrs["description"] = FEATURE_CATALOG.get(col, "")

    print(f"[10_build] HDF5 written: {path} ({len(rows)} events, {len(float_keys)} float cols)")


# ---------------------------------------------------------------------------
# CSV writer (stdlib)
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    all_keys = list(FEATURE_CATALOG.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[10_build] CSV written:  {path} ({len(rows)} events, {len(all_keys)} cols)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build enriched GW event feature table for KAN/PySR symbolic discovery"
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "malda" / "runs" / "feature_table"),
        help="Output directory (default: malda/runs/feature_table/)",
    )
    parser.add_argument(
        "--bbh-only",
        action="store_true",
        help="Only include BBH events (drop BNS/NSBH for cleaner QNM columns)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    # --- Load catalog ---
    if not CATALOG_CSV.exists():
        print(f"[10_build] ERROR: catalog not found at {CATALOG_CSV}", file=sys.stderr)
        return 1
    cat_rows = read_catalog_csv(CATALOG_CSV)
    print(f"[10_build] Loaded {len(cat_rows)} catalog rows from {CATALOG_CSV.name}")

    # --- Load t0 JSON ---
    t0_data: dict[str, Any] = {}
    if T0_JSON.exists():
        t0_data = json.loads(T0_JSON.read_text(encoding="utf-8"))
        print(f"[10_build] Loaded t0 data for {len(t0_data)} events from {T0_JSON.name}")

    # --- Build rows ---
    feature_rows: list[dict] = []
    skipped = 0
    for cat in cat_rows:
        event_id = cat.get("event", "").strip()
        if not event_id:
            continue
        t0_entry = t0_data.get(event_id, {})
        meta = load_metadata(event_id)
        row = build_row(cat, t0_entry, meta)

        # Optionally filter to BBH only
        if args.bbh_only and not row.get("is_bbh"):
            skipped += 1
            continue

        feature_rows.append(row)

    print(f"[10_build] Built {len(feature_rows)} event rows (skipped {skipped})")

    # Summarize NaN counts for key columns
    key_cols = ["m1_src", "m2_src", "Mf", "af", "Q_220", "S_f", "delta_S", "E_rad_frac"]
    for col in key_cols:
        n_valid = sum(1 for r in feature_rows if math.isfinite(float(r.get(col, float("nan")))))
        print(f"  {col:25s}: {n_valid}/{len(feature_rows)} valid")

    # --- Write outputs ---
    write_hdf5(feature_rows, out_dir / "event_features.h5")
    write_csv(feature_rows, out_dir / "event_features.csv")

    # Feature catalog JSON
    catalog_path = out_dir / "feature_catalog.json"
    catalog_path.write_text(
        json.dumps(FEATURE_CATALOG, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[10_build] Feature catalog: {catalog_path}")

    # Summary JSON
    summary = {
        "n_events": len(feature_rows),
        "n_features": len(FEATURE_CATALOG),
        "bbh_only": args.bbh_only,
        "outputs": {
            "hdf5": str(out_dir / "event_features.h5"),
            "csv": str(out_dir / "event_features.csv"),
            "feature_catalog": str(catalog_path),
        },
        "columns": list(FEATURE_CATALOG.keys()),
    }
    (out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"[10_build] Done. {len(feature_rows)} events × {len(FEATURE_CATALOG)} features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
