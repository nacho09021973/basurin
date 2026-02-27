#!/usr/bin/env python3
"""Stage canónico BASURIN s6c: métricas BRUNETE con curvatura PSD."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import require_run_valid, resolve_out_root, validate_run_id, write_json_atomic
from mvp.brunete.core import J0_J1, chi_psd, curvature_KR, psd_log_derivatives_polyfit, sigma
from mvp.contracts import abort, check_inputs, finalize, init_stage

STAGE = "s6c_brunete_psd_curvature"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON inválido en {path}: se esperaba objeto")
    return data


def _pick_psd_payload(run_dir: Path, explicit: Path | None) -> tuple[Path, dict[str, Any]]:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(
                f"PSD no encontrado en ruta explícita: {explicit}. "
                "Comando para regenerar upstream: python mvp/extract_psd.py --run <RUN_ID>."
            )
        return explicit, _read_json(explicit)

    candidates = [
        run_dir / "psd" / "measured_psd.json",
        run_dir / "external_inputs" / "psd_model.json",
    ]
    detected = [str(p) for p in candidates if p.exists()]
    for path in candidates:
        if path.exists():
            return path, _read_json(path)

    expected = run_dir / "external_inputs" / "psd_model.json"
    detected_txt = ", ".join(detected) if detected else "(none)"
    raise FileNotFoundError(
        f"Falta input PSD. Ruta esperada: {expected}. "
        "Comando para regenerar upstream: python mvp/extract_psd.py --run <RUN_ID>. "
        f"Candidatos detectados: {detected_txt}."
    )


def _extract_psd_arrays(psd_payload: dict[str, Any], detector: str) -> tuple[list[float], list[float]]:
    # measured_psd.json helper schema
    if "frequencies_hz" in psd_payload and "psd_values" in psd_payload:
        return psd_payload["frequencies_hz"], psd_payload["psd_values"]

    # minimal deterministic schema with per-detector map
    models = psd_payload.get("models")
    if isinstance(models, dict) and detector in models and isinstance(models[detector], dict):
        item = models[detector]
        return item.get("frequencies_hz", []), item.get("psd_values", [])

    raise ValueError(
        "Schema PSD no reconocido: se requieren frequencies_hz y psd_values "
        "(globales o dentro de models[detector])."
    )


def _to_float(x: Any) -> float | None:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def _compute_payloads(
    *,
    run_id: str,
    run_dir: Path,
    mode: str,
    c_window: float,
    min_points: int,
    sigma_switch: float,
    psd_explicit: Path | None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    estimates_path = run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    if not estimates_path.exists():
        raise FileNotFoundError(f"Missing required inputs: ringdown_estimates: {estimates_path}")

    psd_path, psd_payload = _pick_psd_payload(run_dir, psd_explicit)
    estimates = _read_json(estimates_path)
    event_id = str(estimates.get("event_id", "unknown"))
    per_detector = estimates.get("per_detector")
    if not isinstance(per_detector, dict):
        raise ValueError("Schema inválido: per_detector no está presente en estimates.json")

    metrics_rows: list[dict[str, Any]] = []
    deriv_rows: list[dict[str, Any]] = []
    regime_sigma = Counter()
    regime_chi = Counter()
    n_warnings = 0
    warning_records: list[dict[str, str]] = []

    for detector, det_data in sorted(per_detector.items()):
        warnings: list[str] = []
        if not isinstance(det_data, dict):
            warnings.append("detector_payload_not_object")
            n_warnings += len(warnings)
            metrics_rows.append(
                {
                    "event_id": event_id,
                    "mode": mode,
                    "detector": detector,
                    "warnings": warnings,
                }
            )
            continue

        f_hz = _to_float(det_data.get("f_hz"))
        q = _to_float(det_data.get("Q"))
        tau_s = _to_float(det_data.get("tau_s"))
        rho0 = _to_float(det_data.get("rho0"))
        if rho0 is None:
            rho0 = _to_float(det_data.get("snr_peak"))

        if f_hz is None or q is None or tau_s is None or q <= 0:
            warnings.append("missing_or_invalid_f_Q_tau")
            n_warnings += len(warnings)
            metrics_rows.append(
                {
                    "event_id": event_id,
                    "mode": mode,
                    "detector": detector,
                    "f_hz": f_hz,
                    "Q": q,
                    "tau_s": tau_s,
                    "warnings": warnings,
                }
            )
            continue

        freqs_raw, psd_raw = _extract_psd_arrays(psd_payload, detector)
        try:
            s1, kappa, meta = psd_log_derivatives_polyfit(
                freqs_hz=freqs_raw,
                psd=psd_raw,
                f0_hz=f_hz,
                half_window_hz=c_window,
                min_points=min_points,
            )
        except ValueError as exc:
            if "puntos insuficientes" not in str(exc).lower():
                raise
            warning_records.append(
                {
                    "detector": detector,
                    "code": "PSD_POLYFIT_INSUFFICIENT_POINTS",
                    "detail": str(exc),
                }
            )
            warnings.append("PSD_POLYFIT_INSUFFICIENT_POINTS")
            n_warnings += 1
            metrics_rows.append(
                {
                    "event_id": event_id,
                    "mode": mode,
                    "detector": detector,
                    "f_hz": f_hz,
                    "Q": q,
                    "tau_s": tau_s,
                    "rho0": rho0,
                    "warnings": warnings,
                }
            )
            continue
        sigma_value = sigma(q, kappa)
        chi_value = chi_psd(q, s1, kappa)
        j0, j1, j_meta = J0_J1(sigma_value, sigma_switch=sigma_switch)

        k_val = None
        r_val = None
        if rho0 is not None and rho0 > 0:
            k_val, r_val = curvature_KR(rho0=rho0, Q=q, s1=s1, kappa=kappa)
        else:
            warnings.append("rho0_missing_or_nonpositive")

        regime_sigma_label = (
            "perturbative"
            if abs(sigma_value) < sigma_switch
            else ("closed_form" if sigma_value >= 0 else "not_applicable")
        )
        regime_chi_label = "low" if chi_value < sigma_switch else "elevated"
        regime_sigma[regime_sigma_label] += 1
        regime_chi[regime_chi_label] += 1

        if j_meta.get("status") != "ok":
            warnings.append(f"J0_J1_status={j_meta.get('status')}")

        n_warnings += len(warnings)
        metrics_rows.append(
            {
                "event_id": event_id,
                "mode": mode,
                "detector": detector,
                "f_hz": f_hz,
                "Q": q,
                "tau_s": tau_s,
                "rho0": rho0,
                "s1": s1,
                "kappa": kappa,
                "sigma": sigma_value,
                "chi_psd": chi_value,
                "J0": j0,
                "J1": j1,
                "K": k_val,
                "R": r_val,
                "regime_sigma": regime_sigma_label,
                "regime_chi_psd": regime_chi_label,
                "warnings": warnings,
            }
        )
        deriv_rows.append(
            {
                "event_id": event_id,
                "mode": mode,
                "detector": detector,
                "method": "polyfit_log_psd_deg2",
                "half_window_hz": c_window,
                "n_points": meta["n_points"],
                "polyfit_coefficients": list(meta["poly_coeffs"]),
                "s1": s1,
                "kappa": kappa,
            }
        )

    if not deriv_rows:
        raise RuntimeError("No detector produjo resultado útil en s6c")

    metrics_payload = {
        "schema_version": "brunete_metrics_v1",
        "run_id": run_id,
        "psd_input": str(psd_path.relative_to(run_dir)),
        "metrics": metrics_rows,
    }
    deriv_payload = {
        "schema_version": "psd_derivatives_v1",
        "run_id": run_id,
        "derivatives": deriv_rows,
    }
    summary_stats = {
        "n_rows": len(metrics_rows),
        "n_warnings": n_warnings,
        "regime_sigma_counts": dict(regime_sigma),
        "regime_chi_psd_counts": dict(regime_chi),
    }
    return psd_path, metrics_payload, deriv_payload, summary_stats, warning_records


def main() -> int:
    ap = argparse.ArgumentParser(description="s6c BRUNETE PSD curvature")
    ap.add_argument("--run", required=True)
    ap.add_argument("--c-window", type=float, default=30.0, dest="c_window")
    ap.add_argument("--min-points", type=int, default=7, dest="min_points")
    ap.add_argument("--sigma-switch", type=float, default=0.1, dest="sigma_switch")
    ap.add_argument("--mode", default="220")
    ap.add_argument("--psd-path", default=None, help="Ruta opcional a modelo PSD JSON")
    ap.add_argument("--dry-run", action="store_true", help="Ejecuta cómputo sin escribir artefactos")
    args = ap.parse_args()

    params = {
        "c_window": args.c_window,
        "min_points": args.min_points,
        "sigma_switch": args.sigma_switch,
        "mode": args.mode,
    }
    psd_explicit = Path(args.psd_path).resolve() if args.psd_path else None

    if args.dry_run:
        out_root = resolve_out_root("runs")
        validate_run_id(args.run, out_root)
        require_run_valid(out_root, args.run)
        run_dir = out_root / args.run
        _, metrics_payload, _, _, _ = _compute_payloads(
            run_id=args.run,
            run_dir=run_dir,
            mode=args.mode,
            c_window=args.c_window,
            min_points=args.min_points,
            sigma_switch=args.sigma_switch,
            psd_explicit=psd_explicit,
        )
        for row in metrics_payload["metrics"]:
            print(
                "[dry-run] "
                f"det={row.get('detector')} "
                f"f0_hz={row.get('f_hz')} "
                f"Q={row.get('Q')} "
                f"sigma={row.get('sigma')} "
                f"chi_psd={row.get('chi_psd')} "
                f"regime_sigma={row.get('regime_sigma')} "
                f"regime_chi_psd={row.get('regime_chi_psd')}"
            )
        return 0

    ctx = init_stage(args.run, STAGE, params=params)

    estimates_path = ctx.run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    required = {"ringdown_estimates": estimates_path}
    optional = {"psd_path": psd_explicit} if psd_explicit else None
    check_inputs(ctx, required, optional=optional)

    try:
        psd_path, psd_payload = _pick_psd_payload(ctx.run_dir, psd_explicit)
        check_inputs(ctx, required, optional={"psd_model": psd_path})

        _, metrics_payload, deriv_payload, summary_stats, warning_records = _compute_payloads(
            run_id=args.run,
            run_dir=ctx.run_dir,
            mode=args.mode,
            c_window=args.c_window,
            min_points=args.min_points,
            sigma_switch=args.sigma_switch,
            psd_explicit=psd_explicit,
        )

        metrics_path = ctx.outputs_dir / "brunete_metrics.json"
        deriv_path = ctx.outputs_dir / "psd_derivatives.json"
        write_json_atomic(metrics_path, metrics_payload)
        write_json_atomic(deriv_path, deriv_payload)

        finalize(
            ctx,
            artifacts={
                "brunete_metrics": metrics_path,
                "psd_derivatives": deriv_path,
            },
            results={
                "n_rows": summary_stats["n_rows"],
                "n_warnings": summary_stats["n_warnings"],
            },
            extra_summary={
                "regime_sigma_counts": summary_stats["regime_sigma_counts"],
                "regime_chi_psd_counts": summary_stats["regime_chi_psd_counts"],
                "warnings": warning_records,
                "config": params,
            },
        )

        print(f"OUT_ROOT={ctx.out_root}")
        print(f"STAGE_DIR={ctx.stage_dir}")
        print(f"OUTPUTS_DIR={ctx.outputs_dir}")
        print(f"STAGE_SUMMARY={ctx.stage_dir / 'stage_summary.json'}")
        print(f"MANIFEST={ctx.stage_dir / 'manifest.json'}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
