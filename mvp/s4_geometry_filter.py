#!/usr/bin/env python3
"""MVP Stage 4: Filter compatible geometries from theoretical atlas.

CLI:
    python mvp/s4_geometry_filter.py --run <run_id> --atlas-path atlas.json \
        [--epsilon 0.3]

Inputs:  runs/<run>/s3_ringdown_estimates/outputs/estimates.json + atlas.json
Outputs: runs/<run>/s4_geometry_filter/outputs/compatible_set.json

Method: Euclidean distance in (log f, log Q) space; compatible if d < epsilon.
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

STAGE = "s4_geometry_filter"


def _load_atlas(atlas_path: Path) -> list[dict[str, Any]]:
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
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or "geometry_id" not in e:
            raise ValueError(f"Atlas entry {i}: must be dict with geometry_id")
        if not ("f_hz" in e and "Q" in e) and not isinstance(e.get("phi_atlas"), list):
            raise ValueError(f"Atlas entry {i}: needs (f_hz, Q) or phi_atlas")
    return entries


def compute_compatible_set(
    f_obs: float, Q_obs: float,
    atlas: list[dict[str, Any]], epsilon: float,
) -> dict[str, Any]:
    log_f, log_Q = math.log(f_obs), math.log(Q_obs)
    results: list[dict[str, Any]] = []

    for entry in atlas:
        gid = entry["geometry_id"]
        if "f_hz" in entry and "Q" in entry:
            fa, Qa = float(entry["f_hz"]), float(entry["Q"])
            if fa <= 0 or Qa <= 0:
                continue
            d = math.sqrt((log_f - math.log(fa)) ** 2 + (log_Q - math.log(Qa)) ** 2)
        elif "phi_atlas" in entry:
            phi = entry["phi_atlas"]
            if len(phi) >= 2:
                d = math.sqrt((log_f - phi[0]) ** 2 + (log_Q - phi[1]) ** 2)
            elif len(phi) == 1:
                d = abs(log_Q - phi[0])
            else:
                continue
        else:
            continue
        results.append({"geometry_id": gid, "distance": d,
                        "compatible": d <= epsilon, "metadata": entry.get("metadata")})

    results.sort(key=lambda x: x["distance"])
    compatible = [r for r in results if r["compatible"]]
    n_atlas, n_compat = len(results), len(compatible)
    bits = math.log2(n_atlas / n_compat) if n_atlas > 0 and n_compat > 0 else (
        math.log2(n_atlas) if n_atlas > 0 else 0.0)

    return {
        "schema_version": "mvp_compatible_set_v1",
        "observables": {"f_hz": f_obs, "Q": Q_obs},
        "epsilon": epsilon, "n_atlas": n_atlas, "n_compatible": n_compat,
        "bits_excluded": bits,
        "compatible_geometries": compatible,
        "ranked_all": results[:50],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: filter compatible geometries")
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--epsilon", type=float, default=0.3)
    args = ap.parse_args()

    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    ctx = init_stage(args.run, STAGE, params={"atlas_path": str(atlas_path), "epsilon": args.epsilon})

    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    if not atlas_path.exists():
        abort(ctx, f"Atlas not found: {atlas_path}")
    check_inputs(ctx, {"estimates": estimates_path, "atlas": atlas_path})

    try:
        with open(estimates_path, "r", encoding="utf-8") as f:
            estimates = json.load(f)
        combined = estimates.get("combined", {})
        f_obs, Q_obs = float(combined.get("f_hz", 0)), float(combined.get("Q", 0))
        if f_obs <= 0:
            abort(ctx, f"Invalid f_hz: {f_obs}")
        if Q_obs <= 0:
            abort(ctx, f"Invalid Q: {Q_obs}")

        atlas = _load_atlas(atlas_path)
        result = compute_compatible_set(f_obs, Q_obs, atlas, args.epsilon)
        result["event_id"] = estimates.get("event_id", "unknown")
        result["run_id"] = args.run

        cs_path = ctx.outputs_dir / "compatible_set.json"
        write_json_atomic(cs_path, result)

        finalize(ctx, artifacts={"compatible_set": cs_path}, results={
            "n_atlas": result["n_atlas"], "n_compatible": result["n_compatible"],
            "bits_excluded": result["bits_excluded"],
        })
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
