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
_UNSET = object()


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
    metric: str | None = None,
    metric_params: dict[str, Any] | None = None,
    legacy_labels: bool = False,
    sigma_logf: float | object = _UNSET,
    sigma_logQ: float | object = _UNSET,
    cov_logf_logQ: float | object = _UNSET,
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

    legacy_kwargs_used = any(v is not _UNSET for v in (sigma_logf, sigma_logQ, cov_logf_logQ))
    use_legacy_labels = legacy_labels or (metric is None)

    if metric is None and legacy_kwargs_used:
        s_f = None if sigma_logf is _UNSET else sigma_logf
        s_q = None if sigma_logQ is _UNSET else sigma_logQ
        cov = 0.0 if cov_logf_logQ is _UNSET else cov_logf_logQ
        has_sigma = (s_f is not None) or (s_q is not None)
        has_cov = cov != 0.0
        if has_sigma or has_cov:
            if s_f is None or s_q is None:
                raise ValueError("Non-invertible covariance: sigma_logf and sigma_logQ must be provided together")
            metric_name = "mahalanobis_log"
            metric_params = {
                "sigma_logf": float(s_f),
                "sigma_logQ": float(s_q),
                "cov_logf_logQ": float(cov),
            }
        else:
            metric_name = "euclidean_log"
            metric_params = {}
    else:
        metric_name = metric or "euclidean_log"

    if metric_name not in {"euclidean_log", "mahalanobis_log"}:
        raise ValueError(f"Unsupported metric: {metric_name}")

    params = metric_params or {}
    use_mahalanobis = metric_name == "mahalanobis_log"

    sigma_f_val: float | None = None
    sigma_q_val: float | None = None
    cov_val: float | None = None
    if use_mahalanobis:
        sigma_f_val = params.get("sigma_logf")
        sigma_q_val = params.get("sigma_logQ")
        if "cov_logf_logQ" in params:
            cov_val = params.get("cov_logf_logQ")
        elif "correlation" in params:
            cov_val = float(params["correlation"]) * float(params.get("sigma_logf", 0.0)) * float(params.get("sigma_logQ", 0.0))
        elif "r" in params:
            cov_val = float(params["r"]) * float(params.get("sigma_logf", 0.0)) * float(params.get("sigma_logQ", 0.0))
        elif "rho" in params:
            cov_val = float(params["rho"]) * float(params.get("sigma_logf", 0.0)) * float(params.get("sigma_logQ", 0.0))
        elif "corr_logf_logQ" in params:
            cov_val = float(params["corr_logf_logQ"]) * float(params.get("sigma_logf", 0.0)) * float(params.get("sigma_logQ", 0.0))
        else:
            cov_val = 0.0

        if sigma_f_val is None or sigma_q_val is None:
            raise ValueError("Non-invertible covariance: sigma_logf and sigma_logQ are required")
        sigma_f_val = float(sigma_f_val)
        sigma_q_val = float(sigma_q_val)
        cov_val = float(cov_val)

        if not math.isfinite(sigma_f_val) or sigma_f_val <= 0:
            raise ValueError(
                f"Non-invertible covariance: sigma_logf={sigma_f_val} (must be finite and > 0)")
        if not math.isfinite(sigma_q_val) or sigma_q_val <= 0:
            raise ValueError(
                f"Non-invertible covariance: sigma_logQ={sigma_q_val} (must be finite and > 0)")
        if not math.isfinite(cov_val):
            raise ValueError(
                f"Non-invertible covariance: cov_logf_logQ={cov_val} (must be finite)")

        det = (sigma_f_val * sigma_f_val) * (sigma_q_val * sigma_q_val) - (cov_val * cov_val)
        if det <= 0.0:
            raise ValueError("Non-invertible covariance: det(Σ) <= 0")

        inv_00 = (sigma_q_val * sigma_q_val) / det
        inv_11 = (sigma_f_val * sigma_f_val) / det
        inv_01 = -cov_val / det

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
        "bits_kl": bits,
        "compatible_geometries": compatible,
        "ranked_all": results[:50],
    }

    if use_mahalanobis:
        d2_values = [r["d2"] for r in results]
        out["metric"] = "mahalanobis" if use_legacy_labels else "mahalanobis_log"
        out["threshold_d2"] = epsilon
        out["d2_min"] = min(d2_values) if d2_values else None
        out["covariance_logspace"] = {
            "sigma_logf": sigma_f_val,
            "sigma_logQ": sigma_q_val,
            "cov_logf_logQ": cov_val,
        }
        for item in out["compatible_geometries"]:
            item["d2"] = item.get("d2", item.get("dist2"))
        for item in out["ranked_all"]:
            item["d2"] = item.get("d2", item.get("dist2"))
    else:
        out["metric"] = "euclidean" if use_legacy_labels else "euclidean_log"

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: filter compatible geometries")
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--epsilon", type=float, default=None,
                    help="Threshold: d² for Mahalanobis (default χ²₂(95%%)=5.991), "
                         "or Euclidean distance if no covariance available (default 0.3)")
    ap.add_argument("--metric", choices=["euclidean_log", "mahalanobis_log"], default=None,
                    help="Optional modern metric label override; default keeps legacy labels")
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
        if has_cov:
            if args.metric is None:
                result = compute_compatible_set(
                    f_obs, Q_obs, atlas, threshold,
                    metric=None, legacy_labels=True,
                    sigma_logf=float(sigma_logf_raw),
                    sigma_logQ=float(sigma_logQ_raw),
                    cov_logf_logQ=float(cov_raw),
                )
            else:
                result = compute_compatible_set(
                    f_obs, Q_obs, atlas, threshold,
                    metric=args.metric,
                    metric_params={
                        "sigma_logf": float(sigma_logf_raw),
                        "sigma_logQ": float(sigma_logQ_raw),
                        "cov_logf_logQ": float(cov_raw),
                    },
                )
        else:
            result = compute_compatible_set(
                f_obs, Q_obs, atlas, threshold,
                metric=args.metric,
                legacy_labels=(args.metric is None),
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
    except ValueError as exc:
        abort(ctx, str(exc))
        return 2
    except ZeroDivisionError as exc:
        abort(ctx, f"Non-invertible covariance: {exc}")
        return 2
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
