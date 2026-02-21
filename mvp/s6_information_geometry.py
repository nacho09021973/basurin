#!/usr/bin/env python3
"""MVP Stage 6: Information geometry — conformal metric and curvature in observable space.

CLI:
    python mvp/s6_information_geometry.py --run <run_id> \
        [--psd-model simplified_aligo] [--psd-path /path/to/measured_psd.json]

Inputs:
    runs/<run>/s3_ringdown_estimates/outputs/estimates.json
    runs/<run>/s4_geometry_filter/outputs/compatible_set.json
    [optional] measured_psd.json or .npz (via --psd-path)

Outputs:
    runs/<run>/s6_information_geometry/outputs/curvature.json
    runs/<run>/s6_information_geometry/outputs/metric_diagnostics.json

Method:
    Conformal metric g_ij = Omega(f) * delta_ij in (log f, log Q) space,
    where Omega(f) = snr_peak^2 * S_n(f_ref) / S_n(f).

    Scalar curvature (Ricci scalar in 2D):
        R = -(1/Omega) * laplacian(ln Omega)

    computed via central finite differences in log-f.

    Conformally-weighted distances to atlas entries use trapezoidal
    approximation along the Euclidean path:
        d_conformal = (sqrt(Omega_obs) + sqrt(Omega_atlas)) / 2 * d_euclidean

Caveats (v1):
    - snr_peak is a proxy, not the full Fisher information.
    - The Hilbert estimator is not MLE; covariance is incomplete.
    - Omega depends on f only (not Q) in this version.
    - PSD model is analytic approximation, not measured (use --psd-path for measured).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, check_inputs, finalize, abort
from basurin_io import write_json_atomic

STAGE = "s6_information_geometry"

# Reference frequency for PSD normalization (Hz).
F_REF_HZ = 200.0


# ── PSD models ───────────────────────────────────────────────────────────


def _psd_simplified_aligo(f_hz: float) -> float:
    """Simplified aLIGO noise PSD (analytic approximation).

    Shape based on Ajith et al. 2011 phenomenological fit:
        S_n(f) ~ (f_0/f)^4 + 2 + 2*(f/f_0)^2

    Returns dimensionless shape (not calibrated to strain/Hz).
    """
    x = f_hz / F_REF_HZ
    if x <= 0:
        return float("inf")
    return x ** (-4) + 2.0 + 2.0 * x**2


PSD_MODELS = {
    "simplified_aligo": _psd_simplified_aligo,
}


# ── Measured PSD loader ──────────────────────────────────────────────────


def load_measured_psd(psd_path: str | Path) -> Any:
    """Load a measured PSD from JSON or NPZ file and return a callable.

    Supported formats:
      - JSON: {"frequencies_hz": [...], "psd_values": [...]}
      - NPZ:  arrays "freq" and "psd"

    Returns a callable psd_fn(f_hz: float) -> float that interpolates
    in log-log space. Extrapolates using boundary values.

    Raises ValueError if the file cannot be parsed or arrays are invalid.
    """
    from scipy.interpolate import interp1d

    psd_path = Path(psd_path)
    if not psd_path.exists():
        raise ValueError(f"PSD file not found: {psd_path}")

    suffix = psd_path.suffix.lower()

    if suffix == ".npz":
        data = np.load(psd_path)
        if "freq" not in data or "psd" not in data:
            raise ValueError(f"NPZ must contain 'freq' and 'psd' arrays, got: {list(data.keys())}")
        freqs = np.asarray(data["freq"], dtype=float)
        psd_vals = np.asarray(data["psd"], dtype=float)
    elif suffix in (".json",):
        with open(psd_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        freqs = np.asarray(obj.get("frequencies_hz", []), dtype=float)
        psd_vals = np.asarray(obj.get("psd_values", []), dtype=float)
    else:
        raise ValueError(f"Unsupported PSD file format: {suffix} (use .json or .npz)")

    if freqs.size < 2 or psd_vals.size < 2:
        raise ValueError("PSD arrays must have at least 2 points")
    if freqs.size != psd_vals.size:
        raise ValueError(f"freqs ({freqs.size}) and psd ({psd_vals.size}) length mismatch")

    # Filter to positive frequencies and positive PSD values
    valid = (freqs > 0) & (psd_vals > 0)
    freqs = freqs[valid]
    psd_vals = psd_vals[valid]
    if freqs.size < 2:
        raise ValueError("Too few valid (positive) PSD points after filtering")

    # Sort by frequency
    order = np.argsort(freqs)
    freqs = freqs[order]
    psd_vals = psd_vals[order]

    # Build log-log interpolator
    log_f = np.log10(freqs)
    log_p = np.log10(psd_vals)

    interp = interp1d(
        log_f, log_p,
        kind="linear",
        bounds_error=False,
        fill_value=(log_p[0], log_p[-1]),  # extrapolate with boundary values
    )

    f_min = float(freqs[0])
    f_max = float(freqs[-1])

    def psd_fn(f_hz: float) -> float:
        if f_hz <= 0:
            return float("inf")
        if f_hz < f_min or f_hz > f_max:
            import warnings
            warnings.warn(
                f"PSD queried at f={f_hz:.1f} Hz outside measured range "
                f"[{f_min:.1f}, {f_max:.1f}] Hz; extrapolating boundary value.",
                stacklevel=2,
            )
        return float(10.0 ** interp(math.log10(f_hz)))

    return psd_fn


# ── Conformal factor ────────────────────────────────────────────────────


def conformal_factor(f_hz: float, snr_peak: float, psd_fn: Any) -> float:
    """Omega(f) = snr_peak^2 * S_n(f_ref) / S_n(f).

    Normalized so that Omega(f_ref) = snr_peak^2.
    """
    s_n = psd_fn(f_hz)
    s_n_ref = psd_fn(F_REF_HZ)
    if s_n <= 0 or not math.isfinite(s_n):
        return 0.0
    return snr_peak**2 * s_n_ref / s_n


# ── Curvature ────────────────────────────────────────────────────────────


def scalar_curvature_2d(
    f_obs: float,
    snr_peak: float,
    psd_fn: Any,
    delta_log_f: float = 0.01,
) -> dict[str, float]:
    """Compute Ricci scalar R = -(1/Omega) * laplacian(ln Omega).

    In v1 Omega depends only on f, so:
        laplacian(ln Omega) = d^2(ln Omega)/d(log f)^2

    Uses central finite differences with step delta_log_f.

    Returns dict with R, Omega at observation, and intermediate values
    for diagnostics.
    """
    log_f = math.log(f_obs)
    f_plus = math.exp(log_f + delta_log_f)
    f_minus = math.exp(log_f - delta_log_f)

    omega_center = conformal_factor(f_obs, snr_peak, psd_fn)
    omega_plus = conformal_factor(f_plus, snr_peak, psd_fn)
    omega_minus = conformal_factor(f_minus, snr_peak, psd_fn)

    if omega_center <= 0 or omega_plus <= 0 or omega_minus <= 0:
        return {
            "R": 0.0,
            "omega_obs": omega_center,
            "ln_omega_obs": 0.0,
            "laplacian_ln_omega": 0.0,
            "numerical_valid": False,
        }

    ln_omega_c = math.log(omega_center)
    ln_omega_p = math.log(omega_plus)
    ln_omega_m = math.log(omega_minus)

    # Central second derivative: d^2(ln Omega)/d(log f)^2
    laplacian_ln_omega = (ln_omega_p - 2.0 * ln_omega_c + ln_omega_m) / delta_log_f**2

    R = -laplacian_ln_omega / omega_center

    return {
        "R": R,
        "omega_obs": omega_center,
        "ln_omega_obs": ln_omega_c,
        "laplacian_ln_omega": laplacian_ln_omega,
        "numerical_valid": True,
    }


# ── Conformal distances ─────────────────────────────────────────────────


def conformal_distance(
    f_obs: float,
    Q_obs: float,
    f_atlas: float,
    Q_atlas: float,
    snr_peak: float,
    psd_fn: Any,
) -> dict[str, float]:
    """Compute conformally-weighted distance between observation and atlas entry.

    Euclidean distance in (log f, log Q):
        d_flat = sqrt((log f_obs - log f_a)^2 + (log Q_obs - log Q_a)^2)

    Conformal distance (trapezoidal approximation):
        d_conf = (sqrt(Omega_obs) + sqrt(Omega_atlas)) / 2 * d_flat
    """
    log_f_obs = math.log(f_obs)
    log_Q_obs = math.log(Q_obs)
    log_f_a = math.log(f_atlas)
    log_Q_a = math.log(Q_atlas)

    d_flat = math.sqrt((log_f_obs - log_f_a) ** 2 + (log_Q_obs - log_Q_a) ** 2)

    omega_obs = conformal_factor(f_obs, snr_peak, psd_fn)
    omega_atlas = conformal_factor(f_atlas, snr_peak, psd_fn)

    sqrt_omega_obs = math.sqrt(max(0.0, omega_obs))
    sqrt_omega_atlas = math.sqrt(max(0.0, omega_atlas))

    d_conformal = (sqrt_omega_obs + sqrt_omega_atlas) / 2.0 * d_flat

    return {
        "d_flat": d_flat,
        "d_conformal": d_conformal,
        "omega_obs": omega_obs,
        "omega_atlas": omega_atlas,
    }


# ── Main computation ─────────────────────────────────────────────────────


def compute_information_geometry(
    f_obs: float,
    Q_obs: float,
    snr_peak: float,
    compatible_geometries: list[dict[str, Any]],
    psd_model: str = "simplified_aligo",
    delta_log_f: float = 0.01,
    psd_fn: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full s6 computation.

    Returns (curvature_result, metric_diagnostics).

    Args:
        psd_fn: Optional callable psd_fn(f_hz) -> float. If provided, overrides
                psd_model. Use load_measured_psd() to create from file.
    """
    if psd_fn is None:
        psd_fn = PSD_MODELS[psd_model]

    # Curvature at observed point
    curv = scalar_curvature_2d(f_obs, snr_peak, psd_fn, delta_log_f)

    # Recompute distances for all compatible geometries
    reranked: list[dict[str, Any]] = []
    for geo in compatible_geometries:
        gid = geo["geometry_id"]
        meta = geo.get("metadata")

        # Extract atlas (f, Q) from the geometry entry
        f_a = geo.get("f_hz")
        Q_a = geo.get("Q")

        if f_a is not None and Q_a is not None and f_a > 0 and Q_a > 0:
            dists = conformal_distance(f_obs, Q_obs, f_a, Q_a, snr_peak, psd_fn)
        else:
            dists = {
                "d_flat": geo.get("distance", 0.0),
                "d_conformal": geo.get("distance", 0.0),
                "omega_obs": curv["omega_obs"],
                "omega_atlas": 0.0,
            }

        reranked.append({
            "geometry_id": gid,
            "d_flat": dists["d_flat"],
            "d_conformal": dists["d_conformal"],
            "omega_atlas": dists["omega_atlas"],
            "metadata": meta,
        })

    # Sort by conformal distance
    reranked.sort(key=lambda x: x["d_conformal"])

    # Rank changes relative to flat ordering
    flat_order = sorted(reranked, key=lambda x: x["d_flat"])
    flat_rank = {g["geometry_id"]: i for i, g in enumerate(flat_order)}
    for i, g in enumerate(reranked):
        g["rank_conformal"] = i
        g["rank_flat"] = flat_rank.get(g["geometry_id"], -1)
        g["rank_delta"] = g["rank_flat"] - g["rank_conformal"]

    # Build curvature output
    curvature_result: dict[str, Any] = {
        "schema_version": "mvp_curvature_v1",
        "observables": {"f_hz": f_obs, "Q": Q_obs},
        "snr_peak": snr_peak,
        "psd_model": psd_model,
        "scalar_curvature_R": curv["R"],
        "omega_at_obs": curv["omega_obs"],
        "curvature_numerical_valid": curv["numerical_valid"],
        "n_geometries_reranked": len(reranked),
        "reranked_geometries": reranked,
    }

    # Build diagnostics output
    omega_values = [g["omega_atlas"] for g in reranked if g["omega_atlas"] > 0]
    metric_diagnostics: dict[str, Any] = {
        "schema_version": "mvp_metric_diagnostics_v1",
        "psd_model": psd_model,
        "f_ref_hz": F_REF_HZ,
        "delta_log_f": delta_log_f,
        "omega_at_obs": curv["omega_obs"],
        "ln_omega_at_obs": curv["ln_omega_obs"],
        "laplacian_ln_omega": curv["laplacian_ln_omega"],
        "scalar_curvature_R": curv["R"],
        "numerical_valid": curv["numerical_valid"],
        "omega_range": {
            "min": min(omega_values) if omega_values else 0.0,
            "max": max(omega_values) if omega_values else 0.0,
        },
        "caveats": [
            "snr_peak is a proxy, not full Fisher information",
            "Hilbert estimator is not MLE; covariance incomplete",
            "Omega depends on f only (not Q) in v1",
            "PSD model is analytic approximation, not measured",
        ],
    }

    return curvature_result, metric_diagnostics


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"MVP {STAGE}: conformal metric and curvature in observable space"
    )
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--psd-model",
        choices=list(PSD_MODELS.keys()),
        default="simplified_aligo",
        help="Analytic PSD model for conformal factor (default: simplified_aligo)",
    )
    ap.add_argument(
        "--psd-path",
        default=None,
        help="Path to measured PSD (JSON with frequencies_hz/psd_values, or .npz with freq/psd). "
             "If provided, overrides --psd-model.",
    )
    ap.add_argument(
        "--delta-log-f",
        type=float,
        default=0.01,
        help="Finite-difference step in log-f for curvature (default: 0.01)",
    )
    args = ap.parse_args()

    ctx = init_stage(args.run, STAGE, params={
        "psd_model": args.psd_model,
        "psd_path": args.psd_path,
        "delta_log_f": args.delta_log_f,
    })

    # Resolve inputs
    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    compatible_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"

    check_inputs(ctx, {
        "estimates": estimates_path,
        "compatible_set": compatible_path,
    })

    try:
        # Load estimates
        with open(estimates_path, "r", encoding="utf-8") as f:
            estimates = json.load(f)

        combined = estimates.get("combined", {})
        f_obs = float(combined.get("f_hz", 0))
        Q_obs = float(combined.get("Q", 0))
        if f_obs <= 0:
            abort(ctx, f"Invalid f_hz: {f_obs}")
        if Q_obs <= 0:
            abort(ctx, f"Invalid Q: {Q_obs}")

        # Extract snr_peak: use max across detectors, fall back to combined if absent
        per_det = estimates.get("per_detector", {})
        snr_values = [
            det.get("snr_peak", 0.0)
            for det in per_det.values()
            if isinstance(det, dict) and "snr_peak" in det
        ]
        snr_peak = max(snr_values) if snr_values else combined.get("snr_peak", 1.0)
        if snr_peak <= 0:
            abort(ctx, f"Invalid snr_peak: {snr_peak}")

        # Load compatible set
        with open(compatible_path, "r", encoding="utf-8") as f:
            compat = json.load(f)

        compatible_geometries = compat.get("compatible_geometries", [])
        if not compatible_geometries:
            compatible_geometries = compat.get("ranked_all", [])

        # Resolve PSD function
        measured_psd_fn = None
        psd_label = args.psd_model
        if args.psd_path:
            try:
                measured_psd_fn = load_measured_psd(args.psd_path)
                psd_label = f"measured:{args.psd_path}"
                print(f"[s6] Using measured PSD from: {args.psd_path}", flush=True)
            except ValueError as exc:
                print(f"[s6] WARNING: Could not load measured PSD ({exc}); "
                      f"falling back to {args.psd_model}", flush=True)

        # Compute
        curvature_result, metric_diagnostics = compute_information_geometry(
            f_obs=f_obs,
            Q_obs=Q_obs,
            snr_peak=snr_peak,
            compatible_geometries=compatible_geometries,
            psd_model=args.psd_model,
            delta_log_f=args.delta_log_f,
            psd_fn=measured_psd_fn,
        )

        # Record PSD source in outputs
        curvature_result["psd_source"] = psd_label
        metric_diagnostics["psd_source"] = psd_label

        # Add provenance fields
        curvature_result["event_id"] = estimates.get("event_id", "unknown")
        curvature_result["run_id"] = args.run

        # Write outputs
        curv_path = ctx.outputs_dir / "curvature.json"
        diag_path = ctx.outputs_dir / "metric_diagnostics.json"
        write_json_atomic(curv_path, curvature_result)
        write_json_atomic(diag_path, metric_diagnostics)

        finalize(
            ctx,
            artifacts={"curvature": curv_path, "metric_diagnostics": diag_path},
            results={
                "scalar_curvature_R": curvature_result["scalar_curvature_R"],
                "omega_at_obs": curvature_result["omega_at_obs"],
                "n_geometries_reranked": curvature_result["n_geometries_reranked"],
                "curvature_numerical_valid": curvature_result["curvature_numerical_valid"],
            },
        )
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
