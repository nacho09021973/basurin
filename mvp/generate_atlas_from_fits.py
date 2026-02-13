#!/usr/bin/env python3
"""Generate a QNM atlas from Berti (2009) fitting formulas.

No external dependencies beyond Python stdlib.  Produces an atlas
compatible with s4_geometry_filter and experiment_eps_sweep.

CLI:
    python mvp/generate_atlas_from_fits.py [--out atlas_berti_v2.json]

Grid:
    - M: 10..300 M_sun  (30 log-spaced points)
    - χ: 0.0..0.99      (40 linear points)
    - Modes: (2,2,0), (2,2,1), (3,3,0)
    - Alternatives: EdGB, dCS, Kerr-Newman at representative (M, χ)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.kerr_qnm_fits import (
    BERTI_FITS,
    apply_deviation,
    deviation_dcs,
    deviation_edgb,
    deviation_kerr_newman,
    kerr_qnm,
    make_atlas_entry,
)


# ---------------------------------------------------------------------------
# Grid defaults
# ---------------------------------------------------------------------------
DEFAULT_MASSES = [10, 15, 20, 25, 30, 35, 40, 50, 60, 62, 65, 70, 80,
                  90, 100, 120, 150, 200, 250, 300]
DEFAULT_N_SPINS = 40       # linear grid 0..0.99
DEFAULT_MODES = [(2, 2, 0), (2, 2, 1), (3, 3, 0)]

# Alternative-geometry parameter grids
EDGB_ZETAS = [0.1, 0.3, 0.5, 0.8, 1.0]
DCS_ZETAS = [0.1, 0.3, 0.5, 0.8, 1.0]
KN_CHARGES = [0.1, 0.2, 0.3, 0.5]

# Representative (M, chi) for alternative geometries
ALT_MASSES = [30, 62, 100]
ALT_SPINS = [0.0, 0.3, 0.5, 0.67, 0.8, 0.95]


def generate_kerr_grid(
    masses: list[float],
    n_spins: int,
    modes: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """Generate Kerr atlas entries over M × χ × mode grid."""
    spins = [i * 0.99 / (n_spins - 1) for i in range(n_spins)]
    entries: list[dict[str, Any]] = []

    for mode in modes:
        l, m, n = mode
        for M in masses:
            for chi in spins:
                qnm_result = kerr_qnm(M, chi, mode)
                gid = f"Kerr_M{M:.0f}_a{chi:.4f}_l{l}m{m}n{n}"
                entry = make_atlas_entry(
                    geometry_id=gid,
                    theory="GR_Kerr",
                    qnm=qnm_result,
                    metadata={
                        "family": "kerr",
                        "M_solar": M,
                        "chi": round(chi, 6),
                        "mode": f"({l},{m},{n})",
                        "source": "berti_2009_fit",
                    },
                )
                entries.append(entry)

    return entries


def generate_alternative_entries(
    alt_masses: list[float],
    alt_spins: list[float],
) -> list[dict[str, Any]]:
    """Generate alternative-geometry entries as perturbations on Kerr."""
    entries: list[dict[str, Any]] = []
    mode = (2, 2, 0)

    for M in alt_masses:
        for chi in alt_spins:
            base = kerr_qnm(M, chi, mode)

            # EdGB
            for zeta in EDGB_ZETAS:
                df, dt = deviation_edgb(chi, zeta)
                shifted = apply_deviation(base, df, dt)
                gid = f"EdGB_M{M:.0f}_a{chi:.2f}_z{zeta:.1f}"
                entry = make_atlas_entry(
                    geometry_id=gid,
                    theory="EdGB",
                    qnm=shifted,
                    metadata={
                        "family": "edgb",
                        "M_solar": M,
                        "chi": chi,
                        "zeta": zeta,
                        "delta_f_frac": df,
                        "delta_tau_frac": dt,
                        "mode": "(2,2,0)",
                        "ref": "Blazquez-Salcedo+2016, Maselli+2020",
                    },
                )
                entries.append(entry)

            # dCS (skip chi=0, no effect)
            if chi > 0:
                for zeta in DCS_ZETAS:
                    df, dt = deviation_dcs(chi, zeta)
                    shifted = apply_deviation(base, df, dt)
                    gid = f"dCS_M{M:.0f}_a{chi:.2f}_z{zeta:.1f}"
                    entry = make_atlas_entry(
                        geometry_id=gid,
                        theory="dCS",
                        qnm=shifted,
                        metadata={
                            "family": "dcs",
                            "M_solar": M,
                            "chi": chi,
                            "zeta": zeta,
                            "delta_f_frac": df,
                            "delta_tau_frac": dt,
                            "mode": "(2,2,0)",
                            "ref": "Wagle+2022",
                        },
                    )
                    entries.append(entry)

            # Kerr-Newman
            for q in KN_CHARGES:
                df, dt = deviation_kerr_newman(chi, q)
                shifted = apply_deviation(base, df, dt)
                gid = f"KN_M{M:.0f}_a{chi:.2f}_q{q:.1f}"
                entry = make_atlas_entry(
                    geometry_id=gid,
                    theory="Kerr-Newman",
                    qnm=shifted,
                    metadata={
                        "family": "kerr_newman",
                        "M_solar": M,
                        "chi": chi,
                        "q_charge": q,
                        "delta_f_frac": df,
                        "delta_tau_frac": dt,
                        "mode": "(2,2,0)",
                        "ref": "Dias+2015",
                    },
                )
                entries.append(entry)

    return entries


def build_atlas(
    masses: list[float] | None = None,
    n_spins: int = DEFAULT_N_SPINS,
    modes: list[tuple[int, int, int]] | None = None,
    include_alternatives: bool = True,
) -> dict[str, Any]:
    """Build the full atlas dict."""
    masses = masses or DEFAULT_MASSES
    modes = modes or DEFAULT_MODES

    kerr_entries = generate_kerr_grid(masses, n_spins, modes)

    alt_entries: list[dict[str, Any]] = []
    if include_alternatives:
        alt_entries = generate_alternative_entries(ALT_MASSES, ALT_SPINS)

    all_entries = kerr_entries + alt_entries

    atlas: dict[str, Any] = {
        "schema_version": "basurin_atlas_v2_berti_fits",
        "description": (
            "QNM atlas from Berti et al. (2009) fitting formulas + "
            "parametric beyond-Kerr deviations (EdGB, dCS, Kerr-Newman). "
            "No external dependencies."
        ),
        "provenance": {
            "kerr_fits": "Berti, Cardoso & Starinets (2009) CQG 26, 163001, Table VIII",
            "edgb": "Blazquez-Salcedo+(2016) PRD 94,104024; Maselli+(2020) PRL 124,171101",
            "dcs": "Wagle+(2022) PRD 105, 124003",
            "kerr_newman": "Dias+(2015) PRD 92, 084023",
            "fit_precision": "<5% for chi in [0, 0.99]",
        },
        "grid": {
            "masses_Msun": masses,
            "n_spins": n_spins,
            "chi_range": [0.0, 0.99],
            "modes": [f"({l},{m},{n})" for l, m, n in modes],
        },
        "n_kerr": len(kerr_entries),
        "n_alternative": len(alt_entries),
        "n_total": len(all_entries),
        "entries": all_entries,
    }
    return atlas


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate QNM atlas from Berti fits")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent
                     / "docs" / "ringdown" / "atlas" / "atlas_berti_v2.json"),
        help="Output path for the atlas JSON",
    )
    ap.add_argument("--no-alternatives", action="store_true",
                    help="Skip alternative-geometry entries")
    args = ap.parse_args()

    atlas = build_atlas(include_alternatives=not args.no_alternatives)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(atlas, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"{'=' * 60}")
    print(f"  ATLAS GENERATED — Berti (2009) fits")
    print(f"{'=' * 60}")
    print(f"  Kerr entries:    {atlas['n_kerr']:>6d}")
    print(f"  Alternative:     {atlas['n_alternative']:>6d}")
    print(f"  Total:           {atlas['n_total']:>6d}")
    print(f"  Masses: {atlas['grid']['masses_Msun']}")
    print(f"  Spins:  {atlas['grid']['n_spins']} points in {atlas['grid']['chi_range']}")
    print(f"  Modes:  {atlas['grid']['modes']}")

    f_vals = [e["f_hz"] for e in atlas["entries"]]
    q_vals = [e["Q"] for e in atlas["entries"]]
    print(f"  f range: [{min(f_vals):.1f}, {max(f_vals):.1f}] Hz")
    print(f"  Q range: [{min(q_vals):.2f}, {max(q_vals):.2f}]")
    print(f"  Output:  {out_path}")
    print(f"  Size:    {out_path.stat().st_size / 1024:.1f} KB")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
