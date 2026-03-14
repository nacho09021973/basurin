#!/usr/bin/env python3
"""Build a multimode atlas with canonical geometry IDs.

Input atlas entries in current v2 files are frequently split by mode with IDs
like:
  - Kerr_M62_a0.6600_l2m2n0
  - Kerr_M62_a0.6600_l2m2n1

This tool canonicalizes IDs (strips trailing ``_lXmYnZ``), groups entries by
canonical geometry, and emits one row per physical geometry with mode payloads:
  - mode_220: {f_hz, tau_s, Q}
  - mode_221: {f_hz, tau_s, Q}

Default output is under docs/ringdown/atlas.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import re
from pathlib import Path
from typing import Any

MODE_SUFFIX_RE = re.compile(r"_l(\d+)m(\d+)n(\d+)$")
PHYSICAL_ID_RE = re.compile(r"^(?P<family>[A-Za-z0-9]+)_M(?P<M>[0-9.]+)_a(?P<chi>[0-9.]+)")


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_geometry_id(geometry_id: str) -> str:
    return MODE_SUFFIX_RE.sub("", geometry_id)


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def extract_physical_parameters(entry: dict[str, Any], canonical_id: str) -> dict[str, float]:
    """Extract physically meaningful Kerr-like parameters from metadata or id.

    The v2 atlases usually store dimensionless spin as ``chi`` in metadata,
    while some other generators use ``spin``. Canonical ids also encode
    ``_M..._a...`` where ``a`` is already the dimensionless spin used in this
    repository.
    """
    meta = entry.get("metadata")
    meta_dict = meta if isinstance(meta, dict) else {}

    mass = _coerce_finite_float(meta_dict.get("M_solar"))
    if mass is None:
        mass = _coerce_finite_float(meta_dict.get("M_remnant_Msun"))

    chi = _coerce_finite_float(meta_dict.get("chi"))
    if chi is None:
        chi = _coerce_finite_float(meta_dict.get("spin"))

    match = PHYSICAL_ID_RE.match(canonical_id)
    if match is not None:
        if mass is None:
            mass = float(match.group("M"))
        if chi is None:
            chi = float(match.group("chi"))

    params: dict[str, float] = {}
    if mass is not None:
        params["M_solar"] = mass
    if chi is not None:
        params["chi"] = chi
        params["a_over_m"] = chi
        params["J_over_M2"] = chi

    if chi is not None and abs(chi) <= 1.0 and is_kerr_like(entry, canonical_id):
        root = math.sqrt(1.0 - chi * chi)
        params["kerr_r_plus_over_M"] = 1.0 + root
        params["kerr_area_over_M2"] = 8.0 * math.pi * (1.0 + root)

    return params


def is_kerr_like(entry: dict[str, Any], canonical_id: str) -> bool:
    theory = str(entry.get("theory", ""))
    meta = entry.get("metadata")
    family = str(meta.get("family", "")) if isinstance(meta, dict) else ""
    return theory == "GR_Kerr" or family == "kerr" or canonical_id.startswith("Kerr_")


def validate_physical_parameters(entry: dict[str, Any], canonical_id: str, params: dict[str, float]) -> None:
    """Reject super-extremal Kerr entries at atlas-build time."""
    chi = params.get("chi")
    if chi is None:
        return
    if is_kerr_like(entry, canonical_id) and abs(chi) > 1.0:
        raise ValueError(
            f"Invalid Kerr geometry {canonical_id}: |chi| must be <= 1, got {chi}"
        )


def detect_mode(entry: dict[str, Any]) -> str | None:
    meta = entry.get("metadata")
    mode_str = ""
    if isinstance(meta, dict):
        mode_str = str(meta.get("mode", "")).replace(" ", "")

    if mode_str in {"(2,2,0)", "220"}:
        return "220"
    if mode_str in {"(2,2,1)", "221"}:
        return "221"
    if mode_str in {"(3,3,0)", "330"}:
        return "330"

    gid = str(entry.get("geometry_id", ""))
    m = MODE_SUFFIX_RE.search(gid)
    if m is None:
        return None
    l, mm, n = m.groups()
    if l == "2" and mm == "2" and n == "0":
        return "220"
    if l == "2" and mm == "2" and n == "1":
        return "221"
    if l == "3" and mm == "3" and n == "0":
        return "330"
    return None


def mode_payload(entry: dict[str, Any]) -> dict[str, float]:
    return {
        "f_hz": float(entry["f_hz"]),
        "tau_s": float(entry["tau_s"]),
        "Q": float(entry["Q"]),
    }


def _safe_phi(f_hz: float, q_val: float) -> list[float] | None:
    if f_hz <= 0 or q_val <= 0 or not (math.isfinite(f_hz) and math.isfinite(q_val)):
        return None
    return [math.log(f_hz), math.log(q_val)]


def load_entries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        entries = raw.get("entries")
        if isinstance(entries, list):
            return entries
    raise ValueError(f"Unsupported atlas schema in {path}")


def build_multimode_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for entry in entries:
        gid = entry.get("geometry_id")
        if not isinstance(gid, str) or not gid:
            continue

        canon = canonical_geometry_id(gid)
        mode = detect_mode(entry)

        group = grouped.get(canon)
        if group is None:
            group = {
                "canonical_id": canon,
                "theory": entry.get("theory"),
                "seed_entry": entry,
                "modes": {},
                "source_ids": {},
            }
            grouped[canon] = group

        if mode is None:
            # Keep as fallback under pseudo-mode so the entry is still represented.
            mode = "raw"

        if mode not in group["modes"]:
            group["modes"][mode] = entry
            group["source_ids"][mode] = gid

    out: list[dict[str, Any]] = []
    for canon in sorted(grouped.keys()):
        group = grouped[canon]
        seed = group["seed_entry"]
        modes: dict[str, dict[str, Any]] = group["modes"]
        source_ids: dict[str, str] = group["source_ids"]
        physical = extract_physical_parameters(seed, canon)
        validate_physical_parameters(seed, canon, physical)

        out_entry: dict[str, Any] = {
            "geometry_id": canon,
            "theory": seed.get("theory"),
            "metadata": {},
        }

        seed_meta = seed.get("metadata")
        if isinstance(seed_meta, dict):
            out_entry["metadata"].update(seed_meta)
        out_entry["metadata"].pop("mode", None)
        out_entry["metadata"]["source_geometry_ids"] = source_ids
        out_entry["metadata"]["modes_available"] = sorted([m for m in modes.keys() if m != "raw"])
        if physical:
            out_entry["metadata"]["physical_parameters"] = physical

        for key in ("M_solar", "chi", "a_over_m", "J_over_M2"):
            if key in physical:
                out_entry[key] = physical[key]
        if "kerr_r_plus_over_M" in physical and "kerr_area_over_M2" in physical:
            out_entry["kerr_horizon"] = {
                "r_plus_over_M": physical["kerr_r_plus_over_M"],
                "area_over_M2": physical["kerr_area_over_M2"],
            }

        for mode_key in ("220", "221", "330"):
            e = modes.get(mode_key)
            if e is None:
                continue
            out_entry[f"mode_{mode_key}"] = mode_payload(e)

        # Keep legacy flat fields for compatibility: prefer 220, then 221, then first.
        fallback_mode = None
        for candidate in ("220", "221", "330", "raw"):
            if candidate in modes:
                fallback_mode = candidate
                break
        if fallback_mode is not None:
            e = modes[fallback_mode]
            out_entry["f_hz"] = float(e["f_hz"])
            out_entry["tau_s"] = float(e["tau_s"])
            out_entry["Q"] = float(e["Q"])
            phi = e.get("phi_atlas")
            if isinstance(phi, list) and len(phi) == 2:
                out_entry["phi_atlas"] = [float(phi[0]), float(phi[1])]
            else:
                inferred_phi = _safe_phi(out_entry["f_hz"], out_entry["Q"])
                if inferred_phi is not None:
                    out_entry["phi_atlas"] = inferred_phi

        out.append(out_entry)

    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_in = repo_root / "docs" / "ringdown" / "atlas" / "atlas_berti_v2.json"
    default_out = repo_root / "docs" / "ringdown" / "atlas" / "atlas_berti_v3_multimode.json"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-atlas", dest="in_atlas", type=Path, default=default_in)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument(
        "--out-s4",
        type=Path,
        default=None,
        help="Optional s4-compatible output (default: <out>_s4.json).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_path = args.in_atlas.resolve()
    out_path = args.out.resolve()
    out_s4 = args.out_s4.resolve() if args.out_s4 is not None else out_path.with_name(
        f"{out_path.stem}_s4{out_path.suffix}"
    )

    entries = load_entries(in_path)
    multimode_entries = build_multimode_entries(entries)

    n220 = sum(1 for e in multimode_entries if isinstance(e.get("mode_220"), dict))
    n221 = sum(1 for e in multimode_entries if isinstance(e.get("mode_221"), dict))
    nboth = sum(
        1
        for e in multimode_entries
        if isinstance(e.get("mode_220"), dict) and isinstance(e.get("mode_221"), dict)
    )
    n_phys = sum(1 for e in multimode_entries if "chi" in e and "M_solar" in e)
    n_kerr_horizon = sum(1 for e in multimode_entries if isinstance(e.get("kerr_horizon"), dict))

    payload = {
        "schema_version": "basurin_atlas_v3_multimode",
        "description": "Canonicalized multimode atlas derived from mode-split source atlas.",
        "provenance": {
            "derived_from": str(in_path),
            "derived_utc": _utc_now_iso(),
            "tool": "mvp/tools/generate_multimode_atlas_v3.py",
        },
        "n_source_entries": len(entries),
        "n_total": len(multimode_entries),
        "n_with_mode_220": n220,
        "n_with_mode_221": n221,
        "n_with_both_220_221": nboth,
        "n_with_physical_parameters": n_phys,
        "n_with_kerr_horizon_data": n_kerr_horizon,
        "validation": {
            "kerr_bound_enforced": True,
            "physical_id_pattern": PHYSICAL_ID_RE.pattern,
        },
        "entries": multimode_entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    out_s4.parent.mkdir(parents=True, exist_ok=True)
    out_s4.write_text(json.dumps({"entries": multimode_entries}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"IN_ATLAS={in_path}")
    print(f"OUT_ATLAS={out_path}")
    print(f"OUT_ATLAS_S4={out_s4}")
    print(f"N_SOURCE={len(entries)}")
    print(f"N_TOTAL={len(multimode_entries)}")
    print(f"N_MODE220={n220}")
    print(f"N_MODE221={n221}")
    print(f"N_BOTH={nboth}")
    print(f"N_PHYS={n_phys}")
    print(f"N_KERR_HORIZON={n_kerr_horizon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
