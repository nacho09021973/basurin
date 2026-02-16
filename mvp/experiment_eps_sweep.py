#!/usr/bin/env python3
"""Epsilon-sweep experiment: run s4 geometry filter for multiple epsilon values.

CLI:
    python mvp/experiment_eps_sweep.py --run <run_id> --atlas-path atlas.json \
        [--epsilons 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50]

Requires: runs/<run>/s3_ringdown_estimates/outputs/estimates.json must exist.

Outputs:
    runs/<run>/experiment/ATLAS_REAL_EPS_SWEEP_V1/
        sweep_summary.json          # Full comparison table
        sweep_manifest.json         # SHA256 of all artifacts
        eps_0.050/
            compatible_set.json
            manifest.json
        eps_0.100/
            ...
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

from basurin_io import sha256_file, utc_now_iso, write_json_atomic, write_manifest
from mvp.s4_geometry_filter import _load_atlas, compute_compatible_set

EXPERIMENT_TAG = "ATLAS_REAL_EPS_SWEEP_V1"
DEFAULT_EPSILONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SIGMA_SCALE_TARGETS_D2 = (5.991, 9.210, 11.829)


def suggest_sigma_scale(d2_min: float, target_d2: float) -> float:
    """Return sigma multiplier s needed to move d2_min close to target_d2.

    Since Mahalanobis distance in log-space scales inversely with sigma,
    d2 scales as 1 / s^2 for a uniform sigma scaling. Therefore:

        target_d2 ~= d2_min / s^2  =>  s = sqrt(d2_min / target_d2)
    """
    d2 = float(d2_min)
    target = float(target_d2)
    if not math.isfinite(d2) or d2 <= 0:
        raise ValueError(f"d2_min must be finite and > 0, got {d2}")
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"target_d2 must be finite and > 0, got {target}")
    return math.sqrt(d2 / target)


def _resolve_mahalanobis_params(estimates: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    unc = estimates.get("combined_uncertainty", {}) or {}
    params: dict[str, Any] = {}

    sigma_logf = unc.get("sigma_logf")
    sigma_logQ = unc.get("sigma_logQ")
    cov_logf_logQ = unc.get("cov_logf_logQ")

    if sigma_logf is not None:
        params["sigma_lnf"] = sigma_logf
    if sigma_logQ is not None:
        params["sigma_lnQ"] = sigma_logQ
    if cov_logf_logQ is not None:
        params["cov_logf_logQ"] = cov_logf_logQ

    for key in ("r", "correlation", "rho", "corr_logf_logQ"):
        if key in unc and unc.get(key) is not None:
            params["r"] = unc.get(key)
            break

    if args.sigma_lnf is not None:
        params["sigma_lnf"] = args.sigma_lnf
    if args.sigma_lnQ is not None:
        params["sigma_lnQ"] = args.sigma_lnQ
    if args.correlation is not None:
        params["r"] = args.correlation

    # ── Require sigmas ────────────────────────────────────────────────
    if params.get("sigma_lnf") is None or params.get("sigma_lnQ") is None:
        raise ValueError(
            "mahalanobis_log requires uncertainties: provide --sigma-lnf/--sigma-lnQ or regenerate s3 estimates with combined_uncertainty"
        )

    # ── Ensure correlation is always explicit (never fall through to
    #    DEFAULT_CORRELATION in distance_metrics) ──────────────────────
    if "r" not in params and "cov_logf_logQ" not in params:
        params["r"] = 0.0

    # ── Validate numeric constraints ─────────────────────────────────
    s_lnf = float(params["sigma_lnf"])
    s_lnQ = float(params["sigma_lnQ"])
    if not math.isfinite(s_lnf) or s_lnf <= 0:
        raise ValueError(
            f"Non-invertible covariance: sigma_lnf must be finite and > 0, got {s_lnf}"
        )
    if not math.isfinite(s_lnQ) or s_lnQ <= 0:
        raise ValueError(
            f"Non-invertible covariance: sigma_lnQ must be finite and > 0, got {s_lnQ}"
        )
    r_val = params.get("r")
    if r_val is not None:
        r_val = float(r_val)
        if not math.isfinite(r_val) or abs(r_val) >= 1.0:
            raise ValueError(
                f"Non-invertible covariance: |r| must be < 1, got {r_val}"
            )

    return params


def run_eps_sweep(
    run_id: str,
    atlas_path: Path,
    epsilons: list[float],
    metric: str = "euclidean_log",
    metric_params: dict[str, Any] | None = None,
    runs_root: Path | None = None,
) -> dict[str, Any]:
    """Execute epsilon sweep and write all artifacts.

    Returns the sweep summary dict.
    """
    from basurin_io import resolve_out_root

    out_root = runs_root if runs_root is not None else resolve_out_root("runs")
    run_dir = out_root / run_id

    # ── Validate inputs ──────────────────────────────────────────────
    estimates_path = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    if not estimates_path.exists():
        raise FileNotFoundError(f"s3 estimates not found: {estimates_path}")
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    with open(estimates_path, "r", encoding="utf-8") as f:
        estimates = json.load(f)
    combined = estimates.get("combined", {})
    f_obs = float(combined.get("f_hz", 0))
    Q_obs = float(combined.get("Q", 0))
    if f_obs <= 0 or Q_obs <= 0:
        raise ValueError(f"Invalid observables: f_hz={f_obs}, Q={Q_obs}")

    atlas = _load_atlas(atlas_path)

    # ── Sweep output directory ───────────────────────────────────────
    sweep_dir = run_dir / "experiment" / EXPERIMENT_TAG
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # ── Run sweep ────────────────────────────────────────────────────
    event_id = estimates.get("event_id", "unknown")
    sweep_rows: list[dict[str, Any]] = []
    all_artifacts: dict[str, Path] = {}

    epsilons_sorted = sorted(epsilons)
    for eps in epsilons_sorted:
        eps_label = f"eps_{eps:.3f}"
        eps_dir = sweep_dir / eps_label
        eps_dir.mkdir(parents=True, exist_ok=True)

        result = compute_compatible_set(
            f_obs, Q_obs, atlas, eps,
            metric=metric, metric_params=metric_params,
        )
        result["event_id"] = event_id
        result["run_id"] = run_id

        cs_path = eps_dir / "compatible_set.json"
        write_json_atomic(cs_path, result)

        # Per-epsilon manifest
        write_manifest(eps_dir, {"compatible_set": cs_path})

        # Collect row for summary table
        closest = result["ranked_all"][0] if result["ranked_all"] else None
        sweep_rows.append({
            "epsilon": eps,
            "n_compatible": result["n_compatible"],
            "n_atlas": result["n_atlas"],
            "bits_excluded": round(result["bits_excluded"], 4),
            "bits_kl": round(float(result.get("bits_kl", 0.0)), 4),
            "closest_geometry_id": closest["geometry_id"] if closest else None,
            "closest_distance": round(closest["distance"], 6) if closest else None,
        })
        all_artifacts[eps_label] = cs_path

    # ── Sweep summary ────────────────────────────────────────────────
    summary = {
        "metric": metric,
        "metric_params": dict(metric_params or {}),
        "schema_version": "mvp_eps_sweep_v1",
        "experiment_tag": EXPERIMENT_TAG,
        "created": utc_now_iso(),
        "run_id": run_id,
        "event_id": event_id,
        "observables": {"f_hz": f_obs, "Q": Q_obs},
        "atlas_path": str(atlas_path),
        "atlas_sha256": sha256_file(atlas_path),
        "atlas_n_entries": len(atlas),
        "epsilons": epsilons_sorted,
        "n_epsilons": len(epsilons_sorted),
        "rows": sweep_rows,
        "inputs": {
            "estimates": str(estimates_path),
            "estimates_sha256": sha256_file(estimates_path),
        },
    }
    summary_path = sweep_dir / "sweep_summary.json"
    write_json_atomic(summary_path, summary)
    all_artifacts["sweep_summary"] = summary_path

    # Top-level manifest
    write_manifest(sweep_dir, all_artifacts)

    # ── Console report ───────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  EPSILON SWEEP  —  {EXPERIMENT_TAG}")
    print(f"  run={run_id}  event={event_id}")
    print(f"  f_obs={f_obs:.2f} Hz   Q_obs={Q_obs:.3f}")
    print(f"  atlas: {len(atlas)} entries")
    print(f"{'=' * 64}")
    print(f"  {'eps':>6s}  {'n_compat':>8s}  {'bits_excl':>9s}  {'closest_d':>9s}  closest_id")
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 9}  {'-' * 9}  {'-' * 20}")
    for row in sweep_rows:
        cid = row["closest_geometry_id"] or "—"
        cd = f"{row['closest_distance']:.4f}" if row["closest_distance"] is not None else "—"
        print(
            f"  {row['epsilon']:6.3f}  {row['n_compatible']:8d}  "
            f"{row['bits_excluded']:9.4f}  {cd:>9s}  {cid}"
        )
    print(f"{'=' * 64}")
    print(f"  Results: {sweep_dir}")
    print()

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Epsilon-sweep experiment: run s4 for multiple epsilon values",
    )
    ap.add_argument("--run", required=True, help="Existing run_id with s3 estimates")
    ap.add_argument("--atlas-path", required=True, help="Path to atlas JSON")
    ap.add_argument(
        "--epsilons",
        default=None,
        help="Comma-separated epsilon values (default: 0.05..0.50)",
    )
    ap.add_argument(
        "--metric",
        default="euclidean_log",
        choices=["euclidean_log", "mahalanobis_log"],
        help="Distance metric for compute_compatible_set",
    )
    ap.add_argument("--sigma-lnf", type=float, default=None, help="Sigma for ln(f)")
    ap.add_argument("--sigma-lnQ", type=float, default=None, help="Sigma for ln(Q)")
    ap.add_argument("--correlation", type=float, default=None, help="Correlation r in log-space")
    args = ap.parse_args()

    atlas_path = Path(args.atlas_path)
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    if args.epsilons:
        epsilons = [float(e.strip()) for e in args.epsilons.split(",")]
    else:
        epsilons = DEFAULT_EPSILONS

    try:
        metric_params = None
        if args.metric == "mahalanobis_log":
            from basurin_io import resolve_out_root

            out_root = resolve_out_root("runs")
            estimates_path = out_root / args.run / "s3_ringdown_estimates" / "outputs" / "estimates.json"
            with open(estimates_path, "r", encoding="utf-8") as f:
                estimates = json.load(f)
            metric_params = _resolve_mahalanobis_params(estimates, args)

        run_eps_sweep(args.run, atlas_path, epsilons, metric=args.metric, metric_params=metric_params)

        if args.metric == "mahalanobis_log" and args.sigma_lnf is not None and args.sigma_lnQ is not None:
            from basurin_io import resolve_out_root

            out_root = resolve_out_root("runs")
            sweep_dir = out_root / args.run / "experiment" / EXPERIMENT_TAG
            first_eps = sorted(epsilons)[0]
            cs_path = sweep_dir / f"eps_{first_eps:.3f}" / "compatible_set.json"
            with open(cs_path, "r", encoding="utf-8") as f:
                cs = json.load(f)
            d2_min = float(cs["d2_min"])
            for target_d2 in SIGMA_SCALE_TARGETS_D2:
                scale = suggest_sigma_scale(d2_min, target_d2)
                print(f"sigma_scale_to_target_d2({target_d2:.3f}) = {scale:.6f}")

        return 0
    except ValueError as exc:
        msg = str(exc)
        print(f"ERROR: [experiment_eps_sweep] {msg}", file=sys.stderr)
        if "requires uncertainties" in msg or "Non-invertible covariance" in msg:
            return 2
        return 1
    except Exception as exc:
        print(f"ERROR: [experiment_eps_sweep] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
