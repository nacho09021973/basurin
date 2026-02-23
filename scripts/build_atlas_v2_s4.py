#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SPIN_FROM_BASE_RE = re.compile(r"Kerr_a([0-9]+(?:\.[0-9]+)?)_l2m2n0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build atlas_real_v2_s4.json by adding missing GR deviation points (df=0,dQ=0)."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("docs/ringdown/atlas/atlas_real_v1_s4.json"),
        help="Input v1 atlas path",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path("docs/ringdown/atlas/atlas_real_v2_s4.json"),
        help="Output v2 atlas path",
    )
    return parser.parse_args()


def _extract_spin(entry: dict) -> float | None:
    metadata = entry.get("metadata") or {}
    spin = metadata.get("spin")
    if isinstance(spin, (int, float)):
        return float(spin)

    base_geometry = metadata.get("base_geometry")
    if isinstance(base_geometry, str):
        match = SPIN_FROM_BASE_RE.search(base_geometry)
        if match:
            return float(match.group(1))
    return None


def _is_gr_deviation_entry(entry: dict, spin: float) -> bool:
    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False
    if metadata.get("source") != "parametrized_deviation":
        return False
    if float(metadata.get("spin", float("nan"))) != spin:
        return False
    return (
        float(metadata.get("delta_f_frac", float("nan"))) == 0.0
        and float(metadata.get("delta_Q_frac", float("nan"))) == 0.0
    )




def _format_spin_tag(spin: float) -> str:
    raw = f"{spin:.4f}"
    integer, frac = raw.split(".", 1)
    frac = frac.rstrip("0")
    if len(frac) < 2:
        frac = frac.ljust(2, "0")
    return f"{integer}.{frac}"

def _build_gr_entry(spin: float, base_kerr: dict) -> dict:
    spin_tag = _format_spin_tag(spin)
    base_geometry = f"Kerr_a{spin:.4f}_l2m2n0"
    return {
        "geometry_id": f"bK_GR_a{spin_tag}_df+0.00_dQ+0.00",
        "theory": "beyond_Kerr_df+0.00_dQ+0.00",
        "f_hz": base_kerr.get("f_hz"),
        "tau_s": base_kerr.get("tau_s"),
        "Q": base_kerr.get("Q"),
        "phi_atlas": base_kerr.get("phi_atlas"),
        "metadata": {
            "base_geometry": base_geometry,
            "spin": spin,
            "delta_f_frac": 0.0,
            "delta_Q_frac": 0.0,
            "mode": "(2,2,0)",
            "source": "parametrized_deviation",
            "note": "Fractional shift from Kerr GR prediction",
            "M_remnant_Msun": 62.0,
        },
    }


def main() -> int:
    args = parse_args()

    data = json.loads(args.in_path.read_text(encoding="utf-8"))
    entries = list(data.get("entries", []))

    spins = sorted({s for e in entries if (s := _extract_spin(e)) is not None})
    existing_ids = {e.get("geometry_id") for e in entries}

    kerr_by_spin = {}
    for entry in entries:
        metadata = entry.get("metadata") or {}
        if entry.get("theory") == "GR_Kerr" and isinstance(metadata.get("spin"), (int, float)):
            kerr_by_spin[float(metadata["spin"])] = entry

    for spin in spins:
        if any(_is_gr_deviation_entry(e, spin) for e in entries):
            continue
        base_kerr = kerr_by_spin.get(spin)
        if base_kerr is None:
            continue
        candidate = _build_gr_entry(spin, base_kerr)
        geometry_id = candidate["geometry_id"]
        if geometry_id in existing_ids:
            continue
        entries.append(candidate)
        existing_ids.add(geometry_id)

    out_obj = {"entries": entries}
    serialized = json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n"
    args.out_path.write_text(serialized, encoding="utf-8")

    atlas_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    print(f"atlas_path={args.out_path}")
    print(f"sha256={atlas_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
