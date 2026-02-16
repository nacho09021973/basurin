#!/usr/bin/env python3
"""MVP Stage 4: Filter compatible geometries from theoretical atlas."""
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

from mvp.contracts import abort, check_inputs, finalize, init_stage
from mvp.distance_metrics import (
    euclidean_log,
    get_metric,
    mahalanobis_log,
)
from basurin_io import write_json_atomic

STAGE = "s4_geometry_filter"
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


def _normalize_metric(
    metric: str | None,
    metric_params: dict[str, Any] | None,
    sigma_logf: float | object,
    sigma_logQ: float | object,
    cov_logf_logQ: float | object,
) -> tuple[str, dict[str, Any]]:
    params = dict(metric_params or {})

    legacy_supplied = any(v is not _UNSET for v in (sigma_logf, sigma_logQ, cov_logf_logQ))
    if legacy_supplied:
        if sigma_logf is not _UNSET:
            params.setdefault("sigma_logf", sigma_logf)
        if sigma_logQ is not _UNSET:
            params.setdefault("sigma_logQ", sigma_logQ)
        if cov_logf_logQ is not _UNSET:
            params.setdefault("cov_logf_logQ", cov_logf_logQ)

    metric_name = metric or "euclidean_log"
    if metric is None and any(k in params for k in ("sigma_lnf", "sigma_lnQ", "sigma_logf", "sigma_logQ", "cov_logf_logQ", "r", "rho", "correlation", "corr_logf_logQ")):
        metric_name = "mahalanobis_log"

    if metric_name not in {"euclidean_log", "mahalanobis_log"}:
        raise ValueError(f"Unsupported metric: {metric_name}")

    return metric_name, params


def _coerce_covariance_for_output(metric_params: dict[str, Any]) -> dict[str, float]:
    sigma_lnf = metric_params.get("sigma_lnf", metric_params.get("sigma_logf"))
    sigma_lnQ = metric_params.get("sigma_lnQ", metric_params.get("sigma_logQ"))

    if sigma_lnf is None or sigma_lnQ is None:
        raise ValueError("Non-invertible covariance: sigma_lnf and sigma_lnQ are required")

    sigma_lnf = float(sigma_lnf)
    sigma_lnQ = float(sigma_lnQ)

    r = metric_params.get("r")
    if r is None:
        r = metric_params.get("rho")
    if r is None:
        r = metric_params.get("correlation")
    if r is None:
        r = metric_params.get("corr_logf_logQ")

    cov = metric_params.get("cov_logf_logQ")
    if cov is not None:
        cov = float(cov)
        if not math.isfinite(cov):
            raise ValueError("Non-invertible covariance: cov_logf_logQ must be finite")
    else:
        r = 0.0 if r is None else float(r)
        cov = r * sigma_lnf * sigma_lnQ

    return {
        "sigma_logf": sigma_lnf,
        "sigma_logQ": sigma_lnQ,
        "cov_logf_logQ": cov,
    }


def compute_compatible_set(
    f_obs: float,
    Q_obs: float,
    atlas: list[dict[str, Any]],
    epsilon: float,
    *,
    metric: str = "euclidean_log",
    metric_params: dict[str, Any] | None = None,
    legacy_labels: bool = False,
    sigma_logf: float | object = _UNSET,
    sigma_logQ: float | object = _UNSET,
    cov_logf_logQ: float | object = _UNSET,
) -> dict[str, Any]:
    del legacy_labels  # kept for compatibility with older callers

    if f_obs <= 0 or Q_obs <= 0:
        raise ValueError("Observed f_hz and Q must be > 0")

    metric_name, params = _normalize_metric(metric, metric_params, sigma_logf, sigma_logQ, cov_logf_logQ)

    metric_fn = get_metric(metric_name)
    log_f_obs = math.log(f_obs)
    log_Q_obs = math.log(Q_obs)

    results: list[dict[str, Any]] = []

    for entry in atlas:
        gid = entry["geometry_id"]
        lnf_atlas: float
        lnQ_atlas: float

        if "f_hz" in entry and "Q" in entry:
            fa = float(entry["f_hz"])
            Qa = float(entry["Q"])
            if fa <= 0 or Qa <= 0:
                continue
            lnf_atlas = math.log(fa)
            lnQ_atlas = math.log(Qa)
        elif isinstance(entry.get("phi_atlas"), list) and len(entry["phi_atlas"]) >= 2:
            lnf_atlas = float(entry["phi_atlas"][0])
            lnQ_atlas = float(entry["phi_atlas"][1])
        else:
            continue

        dist = float(metric_fn(log_f_obs, log_Q_obs, lnf_atlas, lnQ_atlas, **params))

        if metric_name == "mahalanobis_log":
            d2 = dist * dist
            results.append(
                {
                    "geometry_id": gid,
                    "distance": dist,
                    "d2": d2,
                    "compatible": d2 <= epsilon,
                    "metadata": entry.get("metadata"),
                }
            )
        else:
            results.append(
                {
                    "geometry_id": gid,
                    "distance": dist,
                    "compatible": dist <= epsilon,
                    "metadata": entry.get("metadata"),
                }
            )

    sort_key = "d2" if metric_name == "mahalanobis_log" else "distance"
    results.sort(key=lambda row: row[sort_key])
    compatible = [row for row in results if row["compatible"]]

    n_atlas = len(results)
    n_compatible = len(compatible)

    bits_excluded = math.log2(n_atlas / n_compatible) if n_atlas > 0 and n_compatible > 0 else (math.log2(n_atlas) if n_atlas > 0 else 0.0)
    # epsilon-independent information baseline (for modern tests)
    bits_kl = math.log2(n_atlas) if n_atlas > 0 else 0.0

    out: dict[str, Any] = {
        "schema_version": "mvp_compatible_set_v1",
        "observables": {"f_hz": f_obs, "Q": Q_obs},
        "metric": metric_name,
        "metric_params": params,
        "epsilon": epsilon,
        "n_atlas": n_atlas,
        "n_compatible": n_compatible,
        "compatible_geometries": compatible,
        "ranked_all": results[:50],
        "bits_excluded": bits_excluded,
        "bits_kl": bits_kl,
    }

    if metric_name == "mahalanobis_log":
        out["threshold_d2"] = epsilon
        out["d2_min"] = min((row["d2"] for row in results), default=None)
        out["covariance_logspace"] = _coerce_covariance_for_output(params)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: filter compatible geometries")
    ap.add_argument("--run", required=True)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--metric", choices=["euclidean_log", "mahalanobis_log"], default=None)
    args = ap.parse_args()

    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    ctx = init_stage(args.run, STAGE, params={"atlas_path": str(atlas_path), "epsilon_cli": args.epsilon})

    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    if not atlas_path.exists():
        abort(ctx, f"Atlas not found: {atlas_path}")
    check_inputs(ctx, {"estimates": estimates_path, "atlas": atlas_path})

    try:
        with open(estimates_path, "r", encoding="utf-8") as f:
            estimates = json.load(f)

        combined = estimates.get("combined", {})
        f_obs = float(combined.get("f_hz", 0.0))
        Q_obs = float(combined.get("Q", 0.0))
        if f_obs <= 0:
            abort(ctx, f"Invalid f_hz: {f_obs}")
        if Q_obs <= 0:
            abort(ctx, f"Invalid Q: {Q_obs}")

        unc = estimates.get("combined_uncertainty", {})
        sigma_logf_raw = unc.get("sigma_logf")
        sigma_logQ_raw = unc.get("sigma_logQ")
        cov_raw = unc.get("cov_logf_logQ")

        has_covariance = sigma_logf_raw is not None and sigma_logQ_raw is not None

        metric_name = args.metric or ("mahalanobis_log" if has_covariance else "euclidean_log")
        if metric_name == "mahalanobis_log":
            threshold = args.epsilon if args.epsilon is not None else CHI2_2DOF_95
            params = {
                "sigma_logf": sigma_logf_raw,
                "sigma_logQ": sigma_logQ_raw,
            }
            if cov_raw is not None:
                params["cov_logf_logQ"] = cov_raw
        else:
            threshold = args.epsilon if args.epsilon is not None else 0.3
            params = {}

        ctx.params["epsilon"] = threshold
        ctx.params["metric"] = metric_name

        atlas = _load_atlas(atlas_path)
        result = compute_compatible_set(
            f_obs,
            Q_obs,
            atlas,
            threshold,
            metric=metric_name,
            metric_params=params,
        )
        result["event_id"] = estimates.get("event_id", "unknown")
        result["run_id"] = args.run

        cs_path = ctx.outputs_dir / "compatible_set.json"
        write_json_atomic(cs_path, result)

        finalize(
            ctx,
            artifacts={"compatible_set": cs_path},
            results={
                "n_atlas": result["n_atlas"],
                "n_compatible": result["n_compatible"],
                "bits_excluded": result["bits_excluded"],
                "metric": result["metric"],
                "threshold_d2": result.get("threshold_d2"),
                "d2_min": result.get("d2_min"),
            },
        )
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
