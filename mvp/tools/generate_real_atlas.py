#!/usr/bin/env python3
"""Generate a real QNM atlas from first principles.

Uses the `qnm` package (Stein 2019, JOSS) to compute exact Kerr QNM
frequencies via Leaver's continued-fraction method, then scales to
physical units for a given remnant mass.

Atlas structure:
  Tier 1 — Kerr GR baseline: sweep spin a ∈ [0, 0.99], mode (2,2,0)
  Tier 2 — Kerr with overtone: same spins, mode (2,2,1)  
  Tier 3 — Parametrized beyond-Kerr deviations at representative spins

Output: JSON file compatible with mvp/s4_geometry_filter.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import qnm

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G_SI = 6.67430e-11       # m^3 kg^-1 s^-2
C_SI = 2.99792458e8      # m/s
MSUN_SI = 1.98892e30     # kg
MSUN_S = G_SI * MSUN_SI / C_SI**3   # M_sun in seconds ≈ 4.9255e-6 s

# ---------------------------------------------------------------------------
# GW150914 remnant parameters (LIGO PE results)
# ---------------------------------------------------------------------------
M_REMNANT_MSUN = 62.0    # Final mass in solar masses
CHI_REMNANT = 0.67       # Final dimensionless spin
M_REMNANT_S = M_REMNANT_MSUN * MSUN_S  # Final mass in seconds


def omega_to_physical(omega_dimless: complex, M_s: float) -> dict:
    """Convert dimensionless QNM ω̃ = M·ω to physical (f_hz, tau_s, Q).
    
    Parameters
    ----------
    omega_dimless : complex
        Dimensionless frequency from qnm package (ω̃ = M·ω).
        Convention: Im(ω̃) < 0 for damped modes.
    M_s : float
        BH mass in seconds.
    
    Returns
    -------
    dict with f_hz, tau_s, Q
    """
    omega_r = omega_dimless.real
    omega_i = omega_dimless.imag  # negative for damped modes
    
    f_hz = omega_r / (2.0 * math.pi * M_s)
    tau_s = -M_s / omega_i  # positive because omega_i < 0
    Q = math.pi * f_hz * tau_s  # = -omega_r / (2 * omega_i)
    
    return {"f_hz": f_hz, "tau_s": tau_s, "Q": Q}


def generate_kerr_atlas(
    spins: np.ndarray,
    M_remnant_s: float,
    modes: list[tuple[int, int, int]] = [(2, 2, 0)],
) -> list[dict]:
    """Generate atlas entries for Kerr BHs at given spins and modes."""
    entries = []
    
    for s_field, l, m, n in [(-2, ll, mm, nn) for ll, mm, nn in modes]:
        print(f"Computing mode (l={l}, m={m}, n={n})...", flush=True)
        mode_seq = qnm.modes_cache(s=s_field, l=l, m=m, n=n)
        
        for a in spins:
            try:
                omega, A, C = mode_seq(a=float(a))
                phys = omega_to_physical(omega, M_remnant_s)
                
                # Dimensionless Φ-space coordinates: (log f, log Q)
                if phys["f_hz"] > 0 and phys["Q"] > 0:
                    phi_atlas = [math.log(phys["f_hz"]), math.log(phys["Q"])]
                else:
                    phi_atlas = None
                
                entry = {
                    "geometry_id": f"Kerr_a{a:.4f}_l{l}m{m}n{n}",
                    "theory": "GR_Kerr",
                    "f_hz": phys["f_hz"],
                    "tau_s": phys["tau_s"],
                    "Q": phys["Q"],
                    "phi_atlas": phi_atlas,
                    "metadata": {
                        "spin": float(a),
                        "mode": f"({l},{m},{n})",
                        "M_remnant_Msun": M_REMNANT_MSUN,
                        "omega_dimless_real": omega.real,
                        "omega_dimless_imag": omega.imag,
                        "source": "qnm_package_v0.4.4_Leaver",
                    },
                }
                entries.append(entry)
            except Exception as exc:
                print(f"  SKIP a={a:.4f} mode ({l},{m},{n}): {exc}", flush=True)
    
    return entries


def generate_beyond_kerr_entries(
    kerr_entries: list[dict],
    representative_spins: list[float],
    delta_f_fracs: list[float],
    delta_Q_fracs: list[float],
) -> list[dict]:
    """Generate parametrized beyond-Kerr entries.
    
    For each representative spin and each (δf/f, δQ/Q) combination,
    create an entry shifted from the Kerr prediction.
    """
    # Index Kerr entries by (spin, mode) for lookup
    kerr_by_spin = {}
    for e in kerr_entries:
        if e["theory"] == "GR_Kerr":
            a = e["metadata"]["spin"]
            mode = e["metadata"]["mode"]
            kerr_by_spin[(round(a, 4), mode)] = e
    
    entries = []
    theory_id = 0
    
    for a_rep in representative_spins:
        key = (round(a_rep, 4), "(2,2,0)")
        if key not in kerr_by_spin:
            continue
        base = kerr_by_spin[key]
        
        for df in delta_f_fracs:
            for dq in delta_Q_fracs:
                if df == 0.0 and dq == 0.0:
                    continue  # skip pure Kerr duplicate
                
                f_new = base["f_hz"] * (1.0 + df)
                Q_new = base["Q"] * (1.0 + dq)
                tau_new = Q_new / (math.pi * f_new) if f_new > 0 else 0
                
                if f_new <= 0 or Q_new <= 0:
                    continue
                
                theory_id += 1
                theory_label = f"beyond_Kerr_df{df:+.2f}_dQ{dq:+.2f}"
                
                entry = {
                    "geometry_id": f"bK_{theory_id:03d}_a{a_rep:.2f}_df{df:+.2f}_dQ{dq:+.2f}",
                    "theory": theory_label,
                    "f_hz": f_new,
                    "tau_s": tau_new,
                    "Q": Q_new,
                    "phi_atlas": [math.log(f_new), math.log(Q_new)],
                    "metadata": {
                        "spin": a_rep,
                        "mode": "(2,2,0)",
                        "base_geometry": base["geometry_id"],
                        "delta_f_frac": df,
                        "delta_Q_frac": dq,
                        "M_remnant_Msun": M_REMNANT_MSUN,
                        "source": "parametrized_deviation",
                        "note": "Fractional shift from Kerr GR prediction",
                    },
                }
                entries.append(entry)
    
    return entries


def main():
    print("=" * 60)
    print("ATLAS REAL: QNM desde primeros principios")
    print("=" * 60)
    print(f"Remnant: M_f = {M_REMNANT_MSUN} M_sun, χ_f = {CHI_REMNANT}")
    print(f"M_f in seconds = {M_REMNANT_S:.6e} s")
    print()
    
    # ── Tier 1: Kerr fundamental mode (2,2,0) ──
    # Dense grid near expected spin, sparser elsewhere
    spins_low = np.linspace(0.0, 0.4, 9)       # 9 points
    spins_mid = np.linspace(0.45, 0.85, 17)     # 17 points (dense near χ_f=0.67)
    spins_high = np.linspace(0.86, 0.99, 14)    # 14 points
    spins = np.unique(np.concatenate([spins_low, spins_mid, spins_high]))
    
    print(f"Spin grid: {len(spins)} values, a ∈ [{spins[0]:.2f}, {spins[-1]:.2f}]")
    
    # Fundamental mode
    kerr_220 = generate_kerr_atlas(spins, M_REMNANT_S, modes=[(2, 2, 0)])
    print(f"Tier 1 (Kerr 220): {len(kerr_220)} entries")
    
    # ── Tier 2: Kerr first overtone (2,2,1) ──
    # Sparser grid — overtone is harder to resolve
    spins_overtone = np.linspace(0.0, 0.99, 20)
    kerr_221 = generate_kerr_atlas(spins_overtone, M_REMNANT_S, modes=[(2, 2, 1)])
    print(f"Tier 2 (Kerr 221): {len(kerr_221)} entries")
    
    # ── Tier 3: Beyond-Kerr parametrized deviations ──
    representative_spins = [0.0, 0.30, 0.50, 0.67, 0.80, 0.95]
    delta_f_fracs = [-0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20]
    delta_Q_fracs = [-0.20, -0.10, 0.0, +0.10, +0.20]
    
    beyond_kerr = generate_beyond_kerr_entries(
        kerr_220, representative_spins, delta_f_fracs, delta_Q_fracs
    )
    print(f"Tier 3 (beyond-Kerr): {len(beyond_kerr)} entries")
    
    # ── Combine ──
    all_entries = kerr_220 + kerr_221 + beyond_kerr
    print(f"\nTotal atlas entries: {len(all_entries)}")
    
    # ── Verify GW150914 entry ──
    gw150914_entry = None
    for e in kerr_220:
        if abs(e["metadata"]["spin"] - CHI_REMNANT) < 0.01:
            gw150914_entry = e
            break
    
    if gw150914_entry:
        print(f"\n{'─' * 40}")
        print(f"GW150914 expected (a={CHI_REMNANT}):")
        print(f"  f = {gw150914_entry['f_hz']:.2f} Hz")
        print(f"  τ = {gw150914_entry['tau_s']*1000:.3f} ms")
        print(f"  Q = {gw150914_entry['Q']:.4f}")
        print(f"  ω̃_real = {gw150914_entry['metadata']['omega_dimless_real']:.10f}")
        print(f"  ω̃_imag = {gw150914_entry['metadata']['omega_dimless_imag']:.10f}")
    
    # ── Write atlas ──
    atlas = {
        "schema_version": "basurin_atlas_v1",
        "description": "Real QNM atlas from first principles (qnm package, Leaver method)",
        "provenance": {
            "package": "qnm v0.4.4 (Stein 2019, JOSS 4:1683)",
            "method": "Leaver continued-fraction (Cook-Zalutskiy spectral angular)",
            "reference": "arXiv:1908.10377",
            "remnant_mass_Msun": M_REMNANT_MSUN,
            "remnant_spin": CHI_REMNANT,
            "event": "GW150914",
        },
        "tiers": {
            "tier1_kerr_220": {"n_entries": len(kerr_220), "description": "Kerr (2,2,0) sweep"},
            "tier2_kerr_221": {"n_entries": len(kerr_221), "description": "Kerr (2,2,1) overtone sweep"},
            "tier3_beyond_kerr": {"n_entries": len(beyond_kerr), "description": "Parametrized beyond-Kerr deviations"},
        },
        "n_total": len(all_entries),
        "phi_space": {
            "coordinates": ["log(f_hz)", "log(Q)"],
            "dimension": 2,
        },
        "entries": all_entries,
    }
    
    out_path = Path("/home/claude/atlas_real_v1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(atlas, f, indent=2, ensure_ascii=False)
    
    print(f"\nAtlas written to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
    
    # ── Summary stats ──
    f_vals = [e["f_hz"] for e in all_entries if e["f_hz"] > 0]
    Q_vals = [e["Q"] for e in all_entries if e["Q"] > 0]
    print(f"\nf_hz range: [{min(f_vals):.1f}, {max(f_vals):.1f}] Hz")
    print(f"Q range:    [{min(Q_vals):.4f}, {max(Q_vals):.4f}]")
    
    # Also write s4-compatible version (just the entries list)
    s4_path = Path("/home/claude/atlas_real_v1_s4.json")
    with open(s4_path, "w", encoding="utf-8") as f:
        json.dump({"entries": all_entries}, f, indent=2, ensure_ascii=False)
    print(f"s4-compatible atlas: {s4_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
