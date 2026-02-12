#!/usr/bin/env python3
"""MVP Stage 4: Filter compatible geometries from theoretical atlas.

CLI:
    python mvp/s4_geometry_filter.py --run <run_id> --atlas-path atlas.json \
        [--epsilon 0.3] [--alpha 1.0]

Inputs:
    runs/<run>/s3_ringdown_estimates/outputs/estimates.json
    atlas.json (external)

Outputs (runs/<run>/s4_geometry_filter/outputs/):
    compatible_set.json   Geometries within epsilon of observed (f, Q)

Method:
    1. Load observed (f, Q) from estimates.json.
    2. For each atlas entry, compute predicted (f, Q) or use phi_atlas coordinates.
    3. Distance = Euclidean in log-space: sqrt(sum((log(obs_i) - log(atlas_i))^2)).
    4. Compatible set = entries with distance < epsilon.

Contracts:
    - Atlas must have >= 1 entry with valid f_hz and Q (or phi_atlas).
    - estimates.json must have combined.f_hz > 0 and combined.Q > 0.
    - Output records n_atlas, n_compatible, epsilon, and ranked geometry list.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "s4_geometry_filter"
UPSTREAM_STAGE = "s3_ringdown_estimates"
EXIT_CONTRACT_FAIL = 2


def _abort(message: str) -> None:
    print(f"[{STAGE_NAME}] ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _load_atlas(atlas_path: Path) -> list[dict[str, Any]]:
    """Load atlas entries from JSON.

    Supports two formats:
      A) List of entries: [{"geometry_id": ..., "f_hz": ..., "Q": ...}, ...]
      B) Object with entries/atlas key: {"entries": [...]} or {"atlas": [...]}

    Each entry MUST have geometry_id, and either:
      - f_hz + Q (direct observables), or
      - phi_atlas (list of floats in phi-space)
    """
    with open(atlas_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("entries") or data.get("atlas") or []
    else:
        raise ValueError(f"Atlas must be a list or object, got {type(data).__name__}")

    if not entries:
        raise ValueError("Atlas has zero entries")

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Atlas entry {i} is not a dict")
        if "geometry_id" not in entry:
            raise ValueError(f"Atlas entry {i} missing geometry_id")
        has_fq = ("f_hz" in entry and "Q" in entry)
        has_phi = ("phi_atlas" in entry and isinstance(entry["phi_atlas"], list))
        if not has_fq and not has_phi:
            raise ValueError(
                f"Atlas entry {i} ({entry['geometry_id']}) must have "
                f"(f_hz, Q) or phi_atlas"
            )

    return entries


def compute_compatible_set(
    f_obs: float,
    Q_obs: float,
    atlas: list[dict[str, Any]],
    epsilon: float,
) -> dict[str, Any]:
    """Compute compatible geometry set by log-space distance.

    For entries with (f_hz, Q):
        d = sqrt((log f_obs - log f_atlas)^2 + (log Q_obs - log Q_atlas)^2)

    For entries with phi_atlas (and no f_hz/Q):
        Requires phi_obs = [log(f_obs), log(Q_obs)] and computes Euclidean distance.

    Returns dict with n_atlas, n_compatible, ranked list.
    """
    log_f_obs = math.log(f_obs)
    log_Q_obs = math.log(Q_obs)

    results: list[dict[str, Any]] = []

    for entry in atlas:
        gid = entry["geometry_id"]

        if "f_hz" in entry and "Q" in entry:
            f_a = float(entry["f_hz"])
            Q_a = float(entry["Q"])
            if f_a <= 0 or Q_a <= 0:
                continue
            d = math.sqrt((log_f_obs - math.log(f_a)) ** 2 + (log_Q_obs - math.log(Q_a)) ** 2)
        elif "phi_atlas" in entry:
            phi = entry["phi_atlas"]
            if len(phi) >= 2:
                d = math.sqrt((log_f_obs - phi[0]) ** 2 + (log_Q_obs - phi[1]) ** 2)
            elif len(phi) == 1:
                # Single-dimensional: compare Q only
                d = abs(log_Q_obs - phi[0])
            else:
                continue
        else:
            continue

        results.append({
            "geometry_id": gid,
            "distance": d,
            "compatible": d <= epsilon,
            "metadata": entry.get("metadata"),
        })

    # Sort by distance (ascending)
    results.sort(key=lambda x: x["distance"])

    compatible = [r for r in results if r["compatible"]]
    n_atlas = len(results)
    n_compatible = len(compatible)

    # Information content
    bits_excluded = 0.0
    if n_atlas > 0 and n_compatible > 0:
        bits_excluded = math.log2(n_atlas / n_compatible)
    elif n_atlas > 0 and n_compatible == 0:
        bits_excluded = math.log2(n_atlas)  # all excluded

    return {
        "schema_version": "mvp_compatible_set_v1",
        "observables": {"f_hz": f_obs, "Q": Q_obs},
        "epsilon": epsilon,
        "n_atlas": n_atlas,
        "n_compatible": n_compatible,
        "bits_excluded": bits_excluded,
        "compatible_geometries": compatible,
        "ranked_all": results[:50],  # Top 50 for inspection
    }


def _write_failure(stage_dir: Path, run_id: str, params: dict, inputs: list, reason: str) -> None:
    summary = {
        "stage": STAGE_NAME, "run": run_id, "created": utc_now_iso(),
        "version": "v1", "parameters": params, "inputs": inputs, "outputs": [],
        "verdict": "FAIL", "error": reason,
    }
    sp = write_stage_summary(stage_dir, summary)
    write_manifest(stage_dir, {"stage_summary": sp}, extra={"verdict": "FAIL", "error": reason})


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE_NAME}: filter compatible geometries")
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-path", required=True, help="Path to theoretical atlas JSON")
    ap.add_argument("--epsilon", type=float, default=0.3, help="Log-space distance threshold")
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = out_root / args.run

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        _abort(str(exc))

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)

    # Validate inputs
    estimates_path = run_dir / UPSTREAM_STAGE / "outputs" / "estimates.json"
    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    if not estimates_path.exists():
        _abort(f"Missing upstream output: {estimates_path}")
    if not atlas_path.exists():
        _abort(f"Atlas file not found: {atlas_path}")

    params: dict[str, Any] = {
        "atlas_path": str(atlas_path),
        "epsilon": args.epsilon,
    }
    inputs_list = [
        {"path": str(estimates_path.relative_to(run_dir)), "sha256": sha256_file(estimates_path)},
        {"path": str(atlas_path), "sha256": sha256_file(atlas_path)},
    ]

    try:
        # Load estimates
        with open(estimates_path, "r", encoding="utf-8") as f:
            estimates = json.load(f)

        combined = estimates.get("combined", {})
        f_obs = float(combined.get("f_hz", 0))
        Q_obs = float(combined.get("Q", 0))

        if f_obs <= 0:
            _abort(f"Invalid f_hz in estimates: {f_obs}")
        if Q_obs <= 0:
            _abort(f"Invalid Q in estimates: {Q_obs}")

        # Load atlas
        atlas = _load_atlas(atlas_path)

        # Compute compatible set
        result = compute_compatible_set(f_obs, Q_obs, atlas, args.epsilon)
        result["event_id"] = estimates.get("event_id", "unknown")
        result["run_id"] = args.run

        # Write output
        cs_path = outputs_dir / "compatible_set.json"
        write_json_atomic(cs_path, result)

        outputs_list = [{"path": str(cs_path.relative_to(run_dir)), "sha256": sha256_file(cs_path)}]

        verdict = "PASS"
        if result["n_compatible"] == 0:
            verdict = "PASS"  # No compatible is still a valid science result

        summary = {
            "stage": STAGE_NAME, "run": args.run, "created": utc_now_iso(),
            "version": "v1", "parameters": params, "inputs": inputs_list,
            "outputs": outputs_list, "verdict": verdict,
            "results": {
                "n_atlas": result["n_atlas"],
                "n_compatible": result["n_compatible"],
                "bits_excluded": result["bits_excluded"],
            },
        }
        sp = write_stage_summary(stage_dir, summary)
        write_manifest(
            stage_dir,
            {"compatible_set": cs_path, "stage_summary": sp},
            extra={"inputs": inputs_list},
        )

        print(
            f"OK: {STAGE_NAME} PASS "
            f"(n_compat={result['n_compatible']}/{result['n_atlas']}, "
            f"bits_excl={result['bits_excluded']:.1f})"
        )
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        _write_failure(stage_dir, args.run, params, inputs_list, str(exc))
        _abort(str(exc))
        return EXIT_CONTRACT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
