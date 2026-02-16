#!/usr/bin/env python3
"""MVP Stage 4b: Spectral curvature diagnostic from QNM overtone atlas.

CLI:
    python mvp/s4b_spectral_curvature.py --run <run_id> --atlas-path atlas.json \
        [--l 2 --m 2] [--C 0.1]

Inputs:  runs/<run>/s3_ringdown_estimates/outputs/estimates.json + atlas.json
Outputs: runs/<run>/s4b_spectral_curvature/outputs/spectral_diagnostics.json

Method: Discrete spectral curvature kappa_n = (w[n+1] - 2*w[n] + w[n-1]) / w[n]
where w_n are complex QNM frequencies for a given (l,m) family across overtones.
Bound rule: |kappa_n| <= C * n^(-2)  (default C=0.1).
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

from mvp.contracts import init_stage, check_inputs, finalize, abort
from basurin_io import write_json_atomic

STAGE = "s4b_spectral_curvature"

DEFAULT_C = 0.1
DEFAULT_BOUND_POWER = 2
DEFAULT_MIN_OVERTONES = 3


# ---------------------------------------------------------------------------
# Atlas loading
# ---------------------------------------------------------------------------

def _load_atlas(atlas_path: Path) -> list[dict[str, Any]]:
    """Load atlas entries from JSON. Supports list or {"entries": [...]}."""
    with open(atlas_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("entries") or data.get("atlas") or []
    else:
        raise ValueError(f"Atlas must be list or object, got {type(data).__name__}")
    if not entries:
        raise ValueError("Atlas has zero entries")
    return entries


def _parse_omega(raw: Any) -> complex:
    """Parse a QNM omega value into a complex number.

    Supports:
        {"re": float, "im": float}
        [re, im]
    """
    if isinstance(raw, dict):
        re_val = raw.get("re")
        im_val = raw.get("im")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        re_val, im_val = raw[0], raw[1]
    else:
        raise ValueError(f"Cannot parse omega: {raw!r}")

    re_f, im_f = float(re_val), float(im_val)
    if not (math.isfinite(re_f) and math.isfinite(im_f)):
        raise ValueError(f"Non-finite omega components: re={re_f}, im={im_f}")
    return complex(re_f, im_f)


def _extract_mode_family(
    entry: dict[str, Any], l_mode: int, m_mode: int,
) -> list[tuple[int, complex]] | None:
    """Extract overtone list [(n, omega), ...] for given (l,m) from an entry.

    Returns None if the entry has no qnm data for this mode family.
    """
    qnm_data = entry.get("qnm")
    if not isinstance(qnm_data, dict):
        return None

    key = f"({l_mode},{m_mode})"
    modes = qnm_data.get(key)
    if not isinstance(modes, list) or not modes:
        return None

    result: list[tuple[int, complex]] = []
    for item in modes:
        if isinstance(item, dict):
            n = int(item.get("n", -1))
            omega_raw = item.get("omega")
            if omega_raw is None or n < 0:
                continue
            omega = _parse_omega(omega_raw)
            result.append((n, omega))

    result.sort(key=lambda x: x[0])
    return result if result else None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_spectral_diagnostics(
    atlas: list[dict[str, Any]],
    l_mode: int,
    m_mode: int,
    C: float,
    bound_power: int = DEFAULT_BOUND_POWER,
    min_overtones: int = DEFAULT_MIN_OVERTONES,
) -> list[dict[str, Any]]:
    """Compute discrete spectral curvature for each atlas geometry.

    Returns a list of diagnostic dicts, one per geometry.
    """
    diagnostics: list[dict[str, Any]] = []

    for entry in atlas:
        gid = entry.get("geometry_id")
        if gid is None:
            raise ValueError(f"Atlas entry missing geometry_id: {entry!r}")

        family = _extract_mode_family(entry, l_mode, m_mode)
        if family is None or len(family) < min_overtones:
            diagnostics.append({
                "geometry_id": gid,
                "n_overtones": len(family) if family else 0,
                "insufficient_overtones": True,
                "kappa": [],
                "kappa_bound": [],
                "max_violation_ratio": 0.0,
                "passes_bound": True,
                "notes": f"Need >= {min_overtones} overtones, have {len(family) if family else 0}",
            })
            continue

        # Build consecutive omega array indexed by overtone number
        n_values = [pair[0] for pair in family]
        omega_map = {pair[0]: pair[1] for pair in family}
        n_min, n_max = min(n_values), max(n_values)

        # Compute kappa for n where n-1, n, n+1 all exist
        kappa_list: list[dict[str, Any]] = []
        kappa_bound_list: list[dict[str, Any]] = []
        max_violation_ratio = 0.0
        passes = True

        for n in range(n_min + 1, n_max):
            if n - 1 not in omega_map or n not in omega_map or n + 1 not in omega_map:
                continue
            w_prev = omega_map[n - 1]
            w_curr = omega_map[n]
            w_next = omega_map[n + 1]

            if abs(w_curr) == 0.0:
                kappa_list.append({
                    "n": n, "re": 0.0, "im": 0.0, "abs": 0.0,
                })
                kappa_bound_list.append({"n": n, "bound": C * n ** (-bound_power)})
                continue

            kappa_val = (w_next - 2.0 * w_curr + w_prev) / w_curr
            kappa_abs = abs(kappa_val)
            bound_val = C * n ** (-bound_power)

            kappa_list.append({
                "n": n,
                "re": kappa_val.real,
                "im": kappa_val.imag,
                "abs": kappa_abs,
            })
            kappa_bound_list.append({"n": n, "bound": bound_val})

            if bound_val > 0:
                ratio = kappa_abs / bound_val
                if ratio > max_violation_ratio:
                    max_violation_ratio = ratio
                if kappa_abs > bound_val:
                    passes = False

        diagnostics.append({
            "geometry_id": gid,
            "n_overtones": len(family),
            "kappa": kappa_list,
            "kappa_bound": kappa_bound_list,
            "max_violation_ratio": max_violation_ratio,
            "passes_bound": passes,
        })

    # Deterministic ordering
    diagnostics.sort(key=lambda d: d["geometry_id"])
    return diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: spectral curvature diagnostic")
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--l", type=int, default=2, dest="l_mode")
    ap.add_argument("--m", type=int, default=2, dest="m_mode")
    ap.add_argument("--C", type=float, default=DEFAULT_C, dest="C_bound")
    args = ap.parse_args()

    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    ctx = init_stage(args.run, STAGE, params={
        "atlas_path": str(atlas_path),
        "l": args.l_mode,
        "m": args.m_mode,
        "C": args.C_bound,
        "bound_power": DEFAULT_BOUND_POWER,
        "min_overtones": DEFAULT_MIN_OVERTONES,
    })

    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    if not atlas_path.exists():
        abort(ctx, f"Atlas not found: {atlas_path}")
    check_inputs(ctx, {"estimates": estimates_path, "atlas": atlas_path})

    try:
        with open(estimates_path, "r", encoding="utf-8") as f:
            estimates = json.load(f)

        atlas = _load_atlas(atlas_path)

        diagnostics = compute_spectral_diagnostics(
            atlas,
            l_mode=args.l_mode,
            m_mode=args.m_mode,
            C=args.C_bound,
        )

        n_total = len(diagnostics)
        n_usable = sum(1 for d in diagnostics if not d.get("insufficient_overtones"))
        violation_ratios = [d["max_violation_ratio"] for d in diagnostics if not d.get("insufficient_overtones")]
        n_passes = sum(1 for d in diagnostics if d.get("passes_bound") and not d.get("insufficient_overtones"))

        output = {
            "schema_version": "mvp_spectral_diagnostics_v1",
            "run_id": args.run,
            "event_id": estimates.get("event_id", "unknown"),
            "mode_family": [args.l_mode, args.m_mode],
            "atlas_path": str(atlas_path),
            "n_geometries_total": n_total,
            "n_geometries_usable": n_usable,
            "diagnostics": diagnostics,
            "parameters": {
                "C": args.C_bound,
                "bound_power": DEFAULT_BOUND_POWER,
                "min_overtones": DEFAULT_MIN_OVERTONES,
            },
        }

        out_path = ctx.outputs_dir / "spectral_diagnostics.json"
        write_json_atomic(out_path, output)

        finalize(ctx, artifacts={"spectral_diagnostics": out_path}, results={
            "n_total": n_total,
            "n_usable": n_usable,
            "C": args.C_bound,
            "mode_family": [args.l_mode, args.m_mode],
            "max_violation_ratio_min": min(violation_ratios) if violation_ratios else None,
            "n_passes": n_passes,
        })
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
