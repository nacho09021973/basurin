#!/usr/bin/env python3
"""MVP Stage 4: Filter compatible geometries from theoretical atlas.

CLI:
    python mvp/s4_geometry_filter.py --run <run_id> --atlas-path atlas.json \
        [--epsilon 5.991]

Inputs:  runs/<run>/s3_ringdown_estimates/outputs/estimates.json + atlas.json
Outputs: runs/<run>/s4_geometry_filter/outputs/compatible_set.json

Method: Mahalanobis distance in (log f, log Q) space using the covariance
matrix from s3 combined_uncertainty.  Compatible if d² ≤ threshold
(default: χ²₂(95%) = 5.991).  Falls back to Euclidean distance when
covariance is unavailable.
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

# χ²(2 dof, 95%) — default threshold for 2D Mahalanobis compatibility
CHI2_2DOF_95 = 5.991


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
    *,
    sigma_logf: float | None = None,
    sigma_logQ: float | None = None,
    cov_logf_logQ: float = 0.0,
) -> dict[str, Any]:
    """Filter atlas geometries by proximity in (log f, log Q) space.

    When *sigma_logf* and *sigma_logQ* are provided, uses Mahalanobis
    distance d² = Δᵀ Σ⁻¹ Δ with *epsilon* as the d² threshold
    (default recommendation: χ²₂(95%) = 5.991).

    When covariance is not provided, falls back to Euclidean distance
    with *epsilon* as the distance threshold (legacy behaviour).

    Raises ``ValueError`` if the covariance matrix is non-invertible
    (FAIL policy for singular Σ).
    """
    log_f, log_Q = math.log(f_obs), math.log(Q_obs)

    use_mahalanobis = sigma_logf is not None and sigma_logQ is not None

    if use_mahalanobis:
        # Validate covariance invertibility (FAIL policy)
        if not math.isfinite(sigma_logf) or sigma_logf <= 0:
            raise ValueError(
                f"Non-invertible covariance: sigma_logf={sigma_logf} "
                f"(must be finite and > 0)")
        if not math.isfinite(sigma_logQ) or sigma_logQ <= 0:
            raise ValueError(
                f"Non-invertible covariance: sigma_logQ={sigma_logQ} "
                f"(must be finite and > 0)")
        if not math.isfinite(cov_logf_logQ):
            raise ValueError(
                f"Non-invertible covariance: cov_logf_logQ={cov_logf_logQ} "
                f"(must be finite)")

        var_f = sigma_logf ** 2
        var_Q = sigma_logQ ** 2
        det = var_f * var_Q - cov_logf_logQ ** 2
        if det <= 0:
            raise ValueError(
                f"Non-invertible covariance: det(Σ)={det:.6e} <= 0 "
                f"(sigma_logf={sigma_logf}, sigma_logQ={sigma_logQ}, "
                f"cov={cov_logf_logQ})")

        # Σ⁻¹ for 2×2: [[var_Q, -cov], [-cov, var_f]] / det
        inv_00 = var_Q / det
        inv_11 = var_f / det
        inv_01 = -cov_logf_logQ / det

    results: list[dict[str, Any]] = []

    for entry in atlas:
        gid = entry["geometry_id"]
        if "f_hz" in entry and "Q" in entry:
            fa, Qa = float(entry["f_hz"]), float(entry["Q"])
            if fa <= 0 or Qa <= 0:
                continue
            dlf = log_f - math.log(fa)
            dlQ = log_Q - math.log(Qa)
        elif "phi_atlas" in entry:
            phi = entry["phi_atlas"]
            if len(phi) >= 2:
                dlf = log_f - phi[0]
                dlQ = log_Q - phi[1]
            elif not use_mahalanobis and len(phi) == 1:
                # 1D fallback only in Euclidean mode
                d = abs(log_Q - phi[0])
                results.append({"geometry_id": gid, "distance": d,
                                "compatible": d <= epsilon,
                                "metadata": entry.get("metadata")})
                continue
            else:
                continue
        else:
            continue

        if use_mahalanobis:
            d2 = dlf * dlf * inv_00 + 2.0 * dlf * dlQ * inv_01 + dlQ * dlQ * inv_11
            results.append({
                "geometry_id": gid, "d2": round(d2, 10),
                "distance": math.sqrt(max(d2, 0.0)),
                "compatible": d2 <= epsilon,
                "metadata": entry.get("metadata"),
            })
        else:
            d = math.sqrt(dlf * dlf + dlQ * dlQ)
            results.append({
                "geometry_id": gid, "distance": d,
                "compatible": d <= epsilon,
                "metadata": entry.get("metadata"),
            })

    sort_key = "d2" if use_mahalanobis else "distance"
    results.sort(key=lambda x: x[sort_key])
    compatible = [r for r in results if r["compatible"]]
    n_atlas, n_compat = len(results), len(compatible)
    bits = math.log2(n_atlas / n_compat) if n_atlas > 0 and n_compat > 0 else (
        math.log2(n_atlas) if n_atlas > 0 else 0.0)

    out: dict[str, Any] = {
        "schema_version": "mvp_compatible_set_v1",
        "observables": {"f_hz": f_obs, "Q": Q_obs},
        "epsilon": epsilon, "n_atlas": n_atlas, "n_compatible": n_compat,
        "bits_excluded": bits,
        "compatible_geometries": compatible,
        "ranked_all": results[:50],
    }

    if use_mahalanobis:
        d2_values = [r["d2"] for r in results]
        out["metric"] = "mahalanobis"
        out["threshold_d2"] = epsilon
        out["d2_min"] = min(d2_values) if d2_values else None
        out["covariance_logspace"] = {
            "sigma_logf": sigma_logf,
            "sigma_logQ": sigma_logQ,
            "cov_logf_logQ": cov_logf_logQ,
        }
    else:
        out["metric"] = "euclidean"

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: filter compatible geometries")
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--epsilon", type=float, default=None,
                    help="Threshold: d² for Mahalanobis (default χ²₂(95%%)=5.991), "
                         "or Euclidean distance if no covariance available (default 0.3)")
    args = ap.parse_args()

    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    ctx = init_stage(args.run, STAGE, params={
        "atlas_path": str(atlas_path),
        "epsilon_cli": args.epsilon,
    })

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

        # Extract covariance from s3 combined_uncertainty
        unc = estimates.get("combined_uncertainty", {})
        sigma_logf_raw = unc.get("sigma_logf")
        sigma_logQ_raw = unc.get("sigma_logQ")
        cov_raw = unc.get("cov_logf_logQ", 0.0)

        has_cov = (sigma_logf_raw is not None and sigma_logQ_raw is not None
                   and isinstance(sigma_logf_raw, (int, float))
                   and isinstance(sigma_logQ_raw, (int, float)))

        if has_cov:
            threshold = args.epsilon if args.epsilon is not None else CHI2_2DOF_95
        else:
            threshold = args.epsilon if args.epsilon is not None else 0.3

        # Update params for traceability
        ctx.params["epsilon"] = threshold
        ctx.params["metric"] = "mahalanobis" if has_cov else "euclidean"

        atlas = _load_atlas(atlas_path)
        result = compute_compatible_set(
            f_obs, Q_obs, atlas, threshold,
            sigma_logf=float(sigma_logf_raw) if has_cov else None,
            sigma_logQ=float(sigma_logQ_raw) if has_cov else None,
            cov_logf_logQ=float(cov_raw) if has_cov else 0.0,
        )
        result["event_id"] = estimates.get("event_id", "unknown")
        result["run_id"] = args.run

        cs_path = ctx.outputs_dir / "compatible_set.json"
        write_json_atomic(cs_path, result)

        finalize(ctx, artifacts={"compatible_set": cs_path}, results={
            "n_atlas": result["n_atlas"], "n_compatible": result["n_compatible"],
            "bits_excluded": result["bits_excluded"],
            "metric": result.get("metric", "unknown"),
            "threshold_d2": result.get("threshold_d2"),
            "d2_min": result.get("d2_min"),
        })
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
