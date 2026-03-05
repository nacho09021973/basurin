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

from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.distance_metrics import (
    euclidean_log,
    get_metric,
    mahalanobis_log,
)
from mvp.schemas import validate_compatible_set
from basurin_io import write_json_atomic
from mvp.path_utils import resolve_run_scoped_input

STAGE = "s4_geometry_filter"
ALLOWED_STAGE_NAMES = {"s4_geometry_filter", "s4_spectral_geometry_filter"}
CHI2_2DOF_95 = 5.991
CHI2_2DOF_95_AUDIT = 5.9915
CHI2_2DOF_997_AUDIT = 11.6183
ALPHA_MAX = 0.80
_UNSET = object()


def _resolve_estimates_path(run_dir: Path, estimates_path_override: str | None) -> Path:
    """Resolve estimates override under run_dir and block traversal/escape."""
    return resolve_run_scoped_input(
        run_dir,
        estimates_path_override,
        default_rel="s3_ringdown_estimates/outputs/estimates.json",
        arg_name="--estimates-path",
    )



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


def _mode_to_string(mode_value: Any) -> str:
    if isinstance(mode_value, (list, tuple)):
        return f"({','.join(str(part) for part in mode_value)})"
    return str(mode_value)


def _filter_atlas_by_mode(
    atlas_entries: list[dict[str, Any]],
    mode_filter: str | None,
) -> list[dict[str, Any]]:
    if mode_filter is None:
        return atlas_entries

    filtered = [
        entry
        for entry in atlas_entries
        if _mode_to_string((entry.get("metadata") or {}).get("mode")) == mode_filter
    ]
    if not filtered:
        raise ValueError(
            f"Atlas filter produced zero entries for mode_filter={mode_filter!r}"
        )
    return filtered


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


def _validate_mahalanobis_params(params: dict[str, Any]) -> None:
    """Validate metric_params for mahalanobis_log (contract-first).

    Raises ValueError("Non-invertible covariance: ...") when sigma values
    are missing, non-finite, or non-positive, or when |r| >= 1.
    """
    sigma_lnf = params.get("sigma_lnf", params.get("sigma_logf"))
    sigma_lnQ = params.get("sigma_lnQ", params.get("sigma_logQ"))

    if sigma_lnf is None or sigma_lnQ is None:
        raise ValueError(
            "Non-invertible covariance: sigma_lnf and sigma_lnQ are required"
        )

    try:
        sigma_lnf = float(sigma_lnf)
        sigma_lnQ = float(sigma_lnQ)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Non-invertible covariance: sigma_lnf and sigma_lnQ must be numeric"
        ) from exc

    if not math.isfinite(sigma_lnf) or sigma_lnf <= 0:
        raise ValueError(
            f"Non-invertible covariance: sigma_lnf must be finite and > 0, got {sigma_lnf}"
        )
    if not math.isfinite(sigma_lnQ) or sigma_lnQ <= 0:
        raise ValueError(
            f"Non-invertible covariance: sigma_lnQ must be finite and > 0, got {sigma_lnQ}"
        )

    for key in ("r", "rho", "correlation", "corr_logf_logQ"):
        r = params.get(key)
        if r is not None:
            try:
                r = float(r)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Non-invertible covariance: correlation must be numeric"
                ) from exc
            if not math.isfinite(r) or abs(r) >= 1.0:
                raise ValueError(
                    f"Non-invertible covariance: |r| must be < 1, got {r}"
                )
            break


def _resolve_fixed_theta_row(
    ranked_rows: list[dict[str, Any]],
    fixed_theta0: str | int | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if fixed_theta0 is None:
        return None

    if isinstance(fixed_theta0, int):
        if 0 <= fixed_theta0 < len(ranked_rows):
            return ranked_rows[fixed_theta0]
        return None

    if isinstance(fixed_theta0, str):
        for row in ranked_rows:
            if row.get("geometry_id") == fixed_theta0:
                return row
        return None

    if isinstance(fixed_theta0, dict):
        gid = fixed_theta0.get("geometry_id")
        if isinstance(gid, str):
            for row in ranked_rows:
                if row.get("geometry_id") == gid:
                    return row
    return None


def _add_mahalanobis_audit_fields(
    out: dict[str, Any],
    *,
    fixed_theta0: str | int | dict[str, Any] | None = None,
    theta0_source: str | None = None,
) -> None:
    ranked_rows = out.get("ranked_all")
    if not isinstance(ranked_rows, list):
        out["atlas_posterior"] = None
        out["chi2_fixed_theta"] = None
        return

    prior_type = "uniform_entries"
    n_rows = len(ranked_rows)
    prior_weight_default = (1.0 / n_rows) if n_rows > 0 else 0.0

    valid_rows: list[dict[str, Any]] = []
    for row in ranked_rows:
        d2_val = row.get("d2")
        if d2_val is None:
            dist = row.get("distance")
            if isinstance(dist, (int, float)) and math.isfinite(dist):
                d2_val = float(dist) * float(dist)

        if isinstance(d2_val, (int, float)) and math.isfinite(d2_val) and d2_val >= 0.0:
            d2_float = float(d2_val)
            row["d2"] = d2_float
            row["log_likelihood_rel"] = -0.5 * d2_float
            row["prior_weight"] = prior_weight_default
            valid_rows.append(row)
        else:
            row["d2"] = row.get("d2") if isinstance(row.get("d2"), (int, float)) else None
            row["log_likelihood_rel"] = None
            row["prior_weight"] = prior_weight_default
            row["delta_lnL"] = None
            row["posterior_weight"] = 0.0

    if valid_rows:
        d2_min = out.get("d2_min")
        if not (isinstance(d2_min, (int, float)) and math.isfinite(d2_min) and d2_min >= 0.0):
            d2_min = min(row["d2"] for row in valid_rows)
            out["d2_min"] = d2_min

        max_ll = max(row["log_likelihood_rel"] for row in valid_rows)
        raw_weights: list[float] = []
        for row in valid_rows:
            lw = row["log_likelihood_rel"] - max_ll
            w = math.exp(lw) * row["prior_weight"]
            raw_weights.append(w)

        norm = sum(raw_weights)
        if norm > 0.0:
            for row, w in zip(valid_rows, raw_weights):
                row["posterior_weight"] = w / norm
                row["delta_lnL"] = _delta_lnL_from_d2(row["d2"], d2_min)
        else:
            for row in valid_rows:
                row["posterior_weight"] = 0.0
                row["delta_lnL"] = _delta_lnL_from_d2(row["d2"], d2_min)

        posterior_sum = sum(float(row.get("posterior_weight", 0.0)) for row in ranked_rows)
        if abs(posterior_sum - 1.0) > 1e-9 and norm > 0.0:
            for row in ranked_rows:
                row["posterior_weight"] = float(row.get("posterior_weight", 0.0)) / posterior_sum

        best_entry_id = ranked_rows[0].get("geometry_id") if ranked_rows else None
        out["atlas_posterior"] = {
            "prior_type": prior_type,
            "normalization": "relative_only",
            "best_entry_id": best_entry_id,
            "log_likelihood_rel_best": -0.5 * float(d2_min),
            "d2_min": d2_min,
            "chi2_interpretation": "min_over_atlas_not_chi2",
        }
    else:
        out["atlas_posterior"] = {
            "prior_type": prior_type,
            "normalization": "relative_only",
            "best_entry_id": ranked_rows[0].get("geometry_id") if ranked_rows else None,
            "log_likelihood_rel_best": None,
            "d2_min": out.get("d2_min"),
            "chi2_interpretation": "min_over_atlas_not_chi2",
        }

    if fixed_theta0 is None:
        out["chi2_fixed_theta"] = None
        return

    theta_row = _resolve_fixed_theta_row(ranked_rows, fixed_theta0)
    theta_d2 = theta_row.get("d2") if isinstance(theta_row, dict) else None
    if not (isinstance(theta_d2, (int, float)) and math.isfinite(theta_d2) and theta_d2 >= 0.0):
        out["chi2_fixed_theta"] = None
        return

    p_value = math.exp(-0.5 * float(theta_d2))
    if theta_d2 <= CHI2_2DOF_95_AUDIT:
        classification = "compatible"
    elif theta_d2 >= CHI2_2DOF_997_AUDIT:
        classification = "excluded"
    else:
        classification = "tension"

    out["chi2_fixed_theta"] = {
        "dof": 2,
        "d2": float(theta_d2),
        "p_value": p_value,
        "classification": classification,
        "compatible_95": CHI2_2DOF_95_AUDIT,
        "excluded_strong": CHI2_2DOF_997_AUDIT,
        "chi2_interpretation": "fixed_theta_valid",
        "theta0_source": theta0_source or "unspecified",
    }


def _quantile_linear(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])

    frac = pos - lo
    return float(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))


def _build_diagnostics(
    *,
    n_atlas: int,
    n_compatible: int,
    ranked_all: list[dict[str, Any]],
    d2_min: float | None,
) -> dict[str, Any]:
    acceptance_fraction = (float(n_compatible) / float(n_atlas)) if n_atlas > 0 else 0.0

    if n_compatible == 0:
        informative_status = "EMPTY"
    elif acceptance_fraction > ALPHA_MAX:
        informative_status = "SATURATED"
    else:
        informative_status = "OK"

    d2_values = sorted(
        float(row["d2"])
        for row in ranked_all
        if isinstance(row.get("d2"), (int, float)) and math.isfinite(float(row["d2"]))
    )

    p10 = _quantile_linear(d2_values, 0.10)
    p25 = _quantile_linear(d2_values, 0.25)
    p50 = _quantile_linear(d2_values, 0.50)
    p75 = _quantile_linear(d2_values, 0.75)
    p90 = _quantile_linear(d2_values, 0.90)

    d2_iqr = (p75 - p25) if (p75 is not None and p25 is not None) else None
    d2_range = (p90 - d2_min) if (p90 is not None and isinstance(d2_min, (int, float))) else None

    return {
        "acceptance_fraction": acceptance_fraction,
        "informative_status": informative_status,
        "alpha_max": ALPHA_MAX,
        "d2_quantiles": {
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
        },
        "d2_iqr": d2_iqr,
        "d2_range": d2_range,
    }


def _delta_lnL_from_d2(d2: float, d2_min: float) -> float:
    return -0.5 * (float(d2) - float(d2_min))


def _resolve_delta_mode_threshold(
    mode_filter: str | None,
    atlas: list[dict[str, Any]],
    *,
    delta_lnL_220: float,
    delta_lnL_221: float,
) -> tuple[float, str]:
    if mode_filter is not None:
        normalized_mode = mode_filter.replace(" ", "")
    else:
        mode_candidates = {
            _mode_to_string((entry.get("metadata") or {}).get("mode")).replace(" ", "")
            for entry in atlas
            if (entry.get("metadata") or {}).get("mode") is not None
        }
        if len(mode_candidates) != 1:
            raise ValueError(
                "threshold_mode=delta_lnL requires filtering a single mode (220 or 221); "
                f"detected modes={sorted(mode_candidates)}"
            )
        normalized_mode = next(iter(mode_candidates))

    if normalized_mode in {"(2,2,0)", "220"}:
        return float(delta_lnL_220), "delta_lnL_220"
    if normalized_mode in {"(2,2,1)", "221"}:
        return float(delta_lnL_221), "delta_lnL_221"

    raise ValueError(
        "threshold_mode=delta_lnL only supports mode 220/221; "
        f"mode_filter={mode_filter!r}"
    )


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
    fixed_theta0: str | int | dict[str, Any] | None = None,
    theta0_source: str | None = None,
    threshold_mode: str = "d2",
    threshold_params: dict[str, Any] | None = None,
    include_ranked_all_full: bool = False,
) -> dict[str, Any]:
    del legacy_labels  # kept for compatibility with older callers

    if f_obs <= 0 or Q_obs <= 0:
        raise ValueError("Observed f_hz and Q must be > 0")

    metric_name, params = _normalize_metric(metric, metric_params, sigma_logf, sigma_logQ, cov_logf_logQ)
    if threshold_mode not in {"d2", "delta_lnL"}:
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
    threshold_params_out = dict(threshold_params or {})

    if metric_name == "mahalanobis_log":
        _validate_mahalanobis_params(params)

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
                    "compatible": False,
                    "f_hz": math.exp(lnf_atlas),
                    "Q": math.exp(lnQ_atlas),
                    "metadata": entry.get("metadata"),
                }
            )
        else:
            results.append(
                {
                    "geometry_id": gid,
                    "distance": dist,
                    "compatible": False,
                    "f_hz": math.exp(lnf_atlas),
                    "Q": math.exp(lnQ_atlas),
                    "metadata": entry.get("metadata"),
                }
            )

    sort_key = "d2" if metric_name == "mahalanobis_log" else "distance"
    results.sort(key=lambda row: row[sort_key])

    likelihood_stats: dict[str, Any] | None = None
    valid_likelihood_rows: list[dict[str, Any]] = []

    if metric_name == "mahalanobis_log":
        for row in results:
            d2_val = row.get("d2")
            if d2_val is None:
                d2_val = row.get("mahalanobis_d2")
            if d2_val is None:
                distance_val = row.get("distance")
                if isinstance(distance_val, (int, float)) and math.isfinite(distance_val):
                    d2_val = float(distance_val) * float(distance_val)

            if isinstance(d2_val, (int, float)) and math.isfinite(d2_val) and d2_val >= 0:
                d2_float = float(d2_val)
                row["d2"] = d2_float
                row["log_likelihood"] = -0.5 * d2_float
                valid_likelihood_rows.append(row)
            else:
                row["log_likelihood"] = None

        if valid_likelihood_rows:
            max_log_likelihood = max(row["log_likelihood"] for row in valid_likelihood_rows)
            d2_min = min(row["d2"] for row in valid_likelihood_rows)
            for row in results:
                ll = row.get("log_likelihood")
                row["delta_log_likelihood"] = ll - max_log_likelihood if isinstance(ll, (int, float)) else None
                if isinstance(row.get("d2"), (int, float)):
                    row["delta_lnL"] = _delta_lnL_from_d2(float(row["d2"]), d2_min)
                else:
                    row["delta_lnL"] = None

            if threshold_mode == "delta_lnL":
                delta_lnL_threshold = threshold_params_out.get("delta_lnL")
                if delta_lnL_threshold is None:
                    raise ValueError("threshold_mode=delta_lnL requires threshold_params['delta_lnL']")
                delta_lnL_threshold = float(delta_lnL_threshold)
                for row in results:
                    row_delta = row.get("delta_lnL")
                    row["compatible"] = isinstance(row_delta, (int, float)) and row_delta >= -delta_lnL_threshold
            else:
                for row in results:
                    row["compatible"] = isinstance(row.get("d2"), (int, float)) and float(row["d2"]) <= epsilon

            best_row = min(valid_likelihood_rows, key=lambda row: row["d2"])
            n_excluded_2sigma = sum(math.sqrt(row["d2"]) > 2.0 for row in valid_likelihood_rows)
            n_excluded_3sigma = sum(math.sqrt(row["d2"]) > 3.0 for row in valid_likelihood_rows)
            likelihood_stats = {
                "metric_type": "gaussian_chi2",
                "max_log_likelihood": max_log_likelihood,
                "best_geometry_id": best_row["geometry_id"],
                "n_excluded_2sigma": n_excluded_2sigma,
                "n_excluded_3sigma": n_excluded_3sigma,
            }
        else:
            for row in results:
                row["delta_log_likelihood"] = None
                row["delta_lnL"] = None
    else:
        for row in results:
            row["log_likelihood"] = None
            row["delta_log_likelihood"] = None
            row["compatible"] = row["distance"] <= epsilon

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
        "likelihood_stats": likelihood_stats,
        "threshold_mode": threshold_mode,
        "threshold_params": threshold_params_out,
    }
    if include_ranked_all_full:
        out["ranked_all_full"] = [dict(row) for row in results]

    if metric_name == "mahalanobis_log":
        d2_min = min((row["d2"] for row in results), default=None)
        distance = None
        if isinstance(d2_min, (int, float)) and math.isfinite(d2_min) and d2_min >= 0:
            distance = float(math.sqrt(d2_min))

        out["threshold_d2"] = epsilon if threshold_mode == "d2" else None
        out["d2_min"] = d2_min
        out["distance"] = distance
        out["threshold_value_effective"] = (
            epsilon if threshold_mode == "d2" else threshold_params_out.get("delta_lnL")
        )
        out["covariance_logspace"] = _coerce_covariance_for_output(params)
        _add_mahalanobis_audit_fields(out, fixed_theta0=fixed_theta0, theta0_source=theta0_source)
    else:
        out["threshold_value_effective"] = epsilon

    out["diagnostics"] = _build_diagnostics(
        n_atlas=n_atlas,
        n_compatible=n_compatible,
        ranked_all=out.get("ranked_all", []),
        d2_min=out.get("d2_min") if isinstance(out.get("d2_min"), (int, float)) else None,
    )

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: filter compatible geometries")
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage-name", default=STAGE)
    ap.add_argument("--atlas-path", required=True)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--threshold-mode", choices=["d2", "delta_lnL"], default="d2")
    ap.add_argument("--delta-lnL-220", type=float, default=0.0)
    ap.add_argument("--delta-lnL-221", type=float, default=0.0)
    ap.add_argument(
        "--mode-filter",
        default=None,
        help='Filter atlas entries by exact metadata.mode string (example: "(2,2,0)")',
    )
    ap.add_argument("--metric", choices=["euclidean_log", "mahalanobis_log"], default=None)
    ap.add_argument(
        "--estimates-path", default=None,
        help="Override path to estimates JSON (default: s3_ringdown_estimates/outputs/estimates.json). "
             "Use to consume spectral or other estimator outputs.",
    )
    args = ap.parse_args()

    if args.stage_name not in ALLOWED_STAGE_NAMES:
        print(
            "ERROR: --stage-name must be one of "
            f"{sorted(ALLOWED_STAGE_NAMES)}; got '{args.stage_name}'",
            file=sys.stderr,
        )
        return 2

    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    ctx = init_stage(args.run, args.stage_name, params={
        "stage_name": args.stage_name,
        "atlas_path": str(atlas_path),
        "epsilon_cli": args.epsilon,
        "threshold_mode": args.threshold_mode,
        "delta_lnL_220": args.delta_lnL_220,
        "delta_lnL_221": args.delta_lnL_221,
        "mode_filter": args.mode_filter,
        "estimates_path_override": args.estimates_path,
    })

    try:
        estimates_path = _resolve_estimates_path(ctx.run_dir, args.estimates_path)
    except ValueError as exc:
        abort(ctx, str(exc))
    if not atlas_path.exists():
        abort(ctx, f"Atlas not found: {atlas_path}")
    default_estimates = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    optional_inputs: dict[str, Path] = {}
    if args.estimates_path and estimates_path != default_estimates:
        optional_inputs["estimates_override_external"] = estimates_path

    check_inputs(
        ctx,
        {"estimates": estimates_path, "atlas": atlas_path},
        optional=optional_inputs or None,
    )

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

        # Guard: s3 may produce sigma=0 when instantaneous frequency has zero MAD
        # (all samples identical). This is a degenerate run, not a code bug — write
        # an empty compatible_set and exit 0 so the pipeline sweep continues.
        if has_covariance:
            _slnf = float(sigma_logf_raw)
            _slnq = float(sigma_logQ_raw)
            if not (math.isfinite(_slnf) and _slnf > 0 and math.isfinite(_slnq) and _slnq > 0):
                _reason = f"Degenerate s3 covariance: sigma_logf={_slnf}, sigma_logQ={_slnq} (MAD=0 fit)"
                degenerate_result: dict[str, Any] = {
                    "n_atlas": 0,
                    "n_compatible": 0,
                    "compatible_entries": [],
                    "metric": "none",
                    "threshold_d2": None,
                    "d2_min": None,
                    "bits_excluded": [],
                    "event_id": estimates.get("event_id", "unknown"),
                    "run_id": args.run,
                    "degenerate": True,
                    "degenerate_reason": _reason,
                }
                cs_path = ctx.outputs_dir / "compatible_set.json"
                write_json_atomic(cs_path, degenerate_result)
                finalize(
                    ctx,
                    artifacts={"compatible_set": cs_path},
                    verdict="FAIL",
                    extra_summary={"error": _reason},
                )
                log_stage_paths(ctx)
                return 0

        if args.delta_lnL_220 < 0 or args.delta_lnL_221 < 0:
            abort(ctx, "delta_lnL thresholds must be >= 0 (--delta-lnL-220, --delta-lnL-221)")

        metric_name = args.metric or ("mahalanobis_log" if has_covariance else "euclidean_log")
        if metric_name == "mahalanobis_log" and not has_covariance:
            abort(
                ctx,
                "mahalanobis_log requires combined_uncertainty in s3 estimates "
                "(sigma_logf and sigma_logQ must be present)",
            )
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

        atlas = _filter_atlas_by_mode(_load_atlas(atlas_path), args.mode_filter)

        threshold_params: dict[str, Any] = {}
        if args.threshold_mode == "delta_lnL":
            if metric_name != "mahalanobis_log":
                abort(ctx, "threshold_mode=delta_lnL requires metric=mahalanobis_log")
            delta_value, delta_source = _resolve_delta_mode_threshold(
                args.mode_filter,
                atlas,
                delta_lnL_220=args.delta_lnL_220,
                delta_lnL_221=args.delta_lnL_221,
            )
            threshold_params = {"delta_lnL": delta_value, "source_flag": delta_source}

        ctx.params["threshold_params"] = threshold_params

        result = compute_compatible_set(
            f_obs,
            Q_obs,
            atlas,
            threshold,
            metric=metric_name,
            metric_params=params,
            threshold_mode=args.threshold_mode,
            threshold_params=threshold_params,
            include_ranked_all_full=True,
        )
        result["event_id"] = estimates.get("event_id", "unknown")
        result["run_id"] = args.run
        result["mode_filter"] = args.mode_filter

        _ok, _errs = validate_compatible_set(
            result,
            strict_mahalanobis=(metric_name == "mahalanobis_log"),
        )
        if not _ok:
            print(
                f"WARNING: compatible_set self-check failed: {_errs}",
                file=sys.stderr,
            )

        cs_path = ctx.outputs_dir / "compatible_set.json"
        ranked_all_full_path = ctx.outputs_dir / "ranked_all_full.json"
        ranked_all_full_rows = result.pop("ranked_all_full", None)
        if not isinstance(ranked_all_full_rows, list):
            abort(ctx, "Failed to build outputs/ranked_all_full.json from in-memory ranked_all")

        write_json_atomic(ranked_all_full_path, ranked_all_full_rows)
        write_json_atomic(cs_path, result)

        finalize(
            ctx,
            artifacts={"compatible_set": cs_path, "ranked_all_full": ranked_all_full_path},
            results={
                "n_atlas": result["n_atlas"],
                "n_compatible": result["n_compatible"],
                "acceptance_fraction": (float(result["n_compatible"]) / float(result["n_atlas"])) if result["n_atlas"] > 0 else 0.0,
                "bits_excluded": result["bits_excluded"],
                "metric": result["metric"],
                "threshold_mode": result.get("threshold_mode"),
                "threshold_params": result.get("threshold_params"),
                "epsilon": result.get("epsilon"),
                "threshold_d2": result.get("threshold_d2"),
                "d2_min": result.get("d2_min"),
                "informative_status": (result.get("diagnostics") or {}).get("informative_status"),
                "d2_quantiles": (result.get("diagnostics") or {}).get("d2_quantiles"),
                "diagnostics": result.get("diagnostics"),
            },
        )
        log_stage_paths(ctx)
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
