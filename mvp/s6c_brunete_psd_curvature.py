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
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "s6c_brunete_psd_curvature"


def extract_psd_from_strain(strain: list[float], fs: float, T_s: float) -> tuple[list[float], list[float]]:
    if fs <= 0 or T_s <= 0:
        raise ValueError("fs y T_s deben ser positivos")
    n = max(16, int(fs * T_s))
    data = [float(x) for x in strain[:n]]
    if not data:
        raise ValueError("strain vacío")
    try:
        from scipy.signal import welch  # type: ignore

        freqs, psd = welch(data, fs=fs, nperseg=min(1024, len(data)), return_onesided=True, scaling="density")
        return [float(f) for f in freqs if 20.0 <= float(f) <= 2048.0], [float(p) for f, p in zip(freqs, psd) if 20.0 <= float(f) <= 2048.0]
    except Exception:
        # fallback deterministic periodogram one-sided
        nfft = len(data)
        two_pi = 2.0 * math.pi
        freqs = []
        vals = []
        for k in range(nfft // 2 + 1):
            re = 0.0
            im = 0.0
            for t, x in enumerate(data):
                ang = two_pi * k * t / nfft
                re += x * math.cos(ang)
                im -= x * math.sin(ang)
            p = (re * re + im * im) / (fs * nfft)
            f = k * fs / nfft
            if 20.0 <= f <= 2048.0:
                freqs.append(float(f))
                vals.append(float(max(p, 1e-30)))
        return freqs, vals


def _interp_linear(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return float((1 - t) * ys[i] + t * ys[i + 1])
    return float(ys[-1])


def compute_psd_conformal_factor(f0: float, tau: float, S_n_func) -> dict[str, float]:
    if f0 <= 0 or tau <= 0:
        raise ValueError("f0/tau inválidos")

    def _h2(f: float, x_f0: float, x_tau: float) -> float:
        w = 1.0 / max(x_tau, 1e-12)
        return 1.0 / (((f - x_f0) ** 2) + w * w)

    fmin = max(20.0, 0.2 * f0)
    fmax = min(2048.0, 3.0 * f0)
    n = 400
    df = (fmax - fmin) / n
    num = 0.0
    den = 0.0
    for i in range(n + 1):
        f = fmin + i * df
        h2 = _h2(f, f0, tau)
        sn = max(float(S_n_func(f)), 1e-30)
        num += h2 / sn
        den += h2
    omega = (num * df) / max(den * df, 1e-30)

    rel = 1e-4
    df0 = max(f0 * rel, 1e-6)
    dt = max(tau * rel, 1e-9)
    # finite-diff without recursion helper
    def _omega(xf0: float, xtau: float) -> float:
        nn = 0.0
        dd = 0.0
        for j in range(n + 1):
            ff = fmin + j * df
            h2 = _h2(ff, xf0, xtau)
            nn += h2 / max(float(S_n_func(ff)), 1e-30)
            dd += h2
        return (nn * df) / max(dd * df, 1e-30)

    domega_df0 = (_omega(f0 + df0, tau) - _omega(f0 - df0, tau)) / (2.0 * df0)
    domega_dtau = (_omega(f0, tau + dt) - _omega(f0, tau - dt)) / (2.0 * dt)
    return {"omega": float(omega), "domega_df0": float(domega_df0), "domega_dtau": float(domega_dtau)}


def brunete_fisher_metric(f0: float, tau: float, S_n_func) -> list[list[float]]:
    cf = compute_psd_conformal_factor(f0, tau, S_n_func)
    omega = cf["omega"]
    q = max(math.pi * f0 * tau, 1e-8)
    o1q2 = 1.0 / (q * q)
    # metric analytic surrogate with O(1/Q^2) corrections
    g11 = omega * (1.0 / (f0 * f0)) * (1.0 + 0.5 * o1q2)
    g22 = omega * (1.0 / (tau * tau)) * (1.0 + 1.5 * o1q2)
    g12 = omega * (0.25 / (f0 * tau)) * o1q2
    return [[float(g11), float(g12)], [float(g12), float(g22)]]


def detect_psd_contamination(curvature_analytic: list[list[float]], curvature_psd: list[list[float]]) -> dict[str, Any]:
    tr_a = curvature_analytic[0][0] + curvature_analytic[1][1]
    tr_p = curvature_psd[0][0] + curvature_psd[1][1]
    omega_ratio = float(tr_p / tr_a) if tr_a != 0 else 1.0
    flag = "PSD_DOMINATED" if abs(omega_ratio - 1.0) > 0.15 else "PSD_CLEAN"
    return {"ratio": omega_ratio, "flag": flag}


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
    chi_psd_threshold: float,
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
        regime_chi_label = "low" if chi_value < chi_psd_threshold else "elevated"
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
    ap.add_argument("--chi-psd-threshold", type=float, default=1.0, dest="chi_psd_threshold")
    ap.add_argument("--mode", default="220")
    ap.add_argument("--psd-path", default=None, help="Ruta opcional a modelo PSD JSON")
    ap.add_argument("--dry-run", action="store_true", help="Ejecuta cómputo sin escribir artefactos")
    args = ap.parse_args()

    params = {
        "c_window": args.c_window,
        "min_points": args.min_points,
        "sigma_switch": args.sigma_switch,
        "chi_psd_threshold": args.chi_psd_threshold,
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
            chi_psd_threshold=args.chi_psd_threshold,
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
            chi_psd_threshold=args.chi_psd_threshold,
            psd_explicit=psd_explicit,
        )

        first = next((r for r in metrics_payload.get("metrics", []) if isinstance(r, dict) and r.get("f_hz") and r.get("tau_s")), None)
        if first is None:
            raise RuntimeError("No hay fila con f_hz/tau_s para construir curvatura PSD")
        f0 = float(first["f_hz"])
        tau = float(first["tau_s"])
        freqs_raw, psd_raw = _extract_psd_arrays(psd_payload, str(first.get("detector", "H1")))
        if not freqs_raw or not psd_raw:
            raise RuntimeError("PSD vacío para construir curvatura")

        def s_n_func(f: float) -> float:
            return _interp_linear([float(x) for x in freqs_raw], [float(y) for y in psd_raw], float(f))

        analytic = [[1.0 / (f0 * f0), 0.0], [0.0, 1.0 / (tau * tau)]]
        psd_metric = brunete_fisher_metric(f0, tau, s_n_func)
        contam = detect_psd_contamination(analytic, psd_metric)
        ev_trace = psd_metric[0][0] + psd_metric[1][1]
        ev_det = psd_metric[0][0] * psd_metric[1][1] - psd_metric[0][1] * psd_metric[1][0]
        disc = max(ev_trace * ev_trace - 4.0 * ev_det, 0.0)
        l1 = 0.5 * (ev_trace + math.sqrt(disc))
        l2 = 0.5 * (ev_trace - math.sqrt(disc))
        omega = compute_psd_conformal_factor(f0, tau, s_n_func)["omega"]
        curvature_payload = {
            "schema_version": "brunete_curvature_v1",
            "omega_conformal_factor": float(omega),
            "psd_contamination_flag": contam["flag"],
            "curvature_analytic_2x2": analytic,
            "curvature_psd_2x2": psd_metric,
            "principal_curvatures_psd": [float(l1), float(l2)],
            "ranking_score_psd": float(min(l1, l2)),
        }

        metrics_path = ctx.outputs_dir / "brunete_metrics.json"
        deriv_path = ctx.outputs_dir / "psd_derivatives.json"
        curvature_path = ctx.outputs_dir / "brunete_curvature.json"
        write_json_atomic(metrics_path, metrics_payload)
        write_json_atomic(deriv_path, deriv_payload)
        write_json_atomic(curvature_path, curvature_payload)

        finalize(
            ctx,
            artifacts={
                "brunete_metrics": metrics_path,
                "psd_derivatives": deriv_path,
                "brunete_curvature": curvature_path,
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

        log_stage_paths(ctx)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
